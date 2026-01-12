#!/usr/bin/env python3
"""Bulk upload PDFs for a new country collection while skipping existing entries."""

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from typing import Iterable, Set

from sqlalchemy.orm import Session

from database import SessionLocal
from models import PDF, User

try:
    from app import process_file_in_background, get_file_type
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import application helpers: {exc}")

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload all missing PDFs from a country folder into the vector store."
    )
    parser.add_argument("folder", help="Path to the folder whose name represents the country")
    user_id_group = parser.add_mutually_exclusive_group(required=True)
    user_id_group.add_argument("--user-id", type=int, help="Numeric user ID that owns the PDFs")
    user_id_group.add_argument("--username", help="Username that owns the PDFs")
    parser.add_argument(
        "--description",
        help="Optional description applied to each uploaded PDF",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list PDFs that would be uploaded without performing the operation",
    )
    return parser.parse_args()


def resolve_user_id(args: argparse.Namespace) -> int:
    session: Session = SessionLocal()
    try:
        if args.user_id is not None:
            user = session.query(User).filter(User.id == args.user_id).first()
        else:
            user = session.query(User).filter(User.username == args.username).first()
        if not user:
            lookup = args.user_id if args.user_id is not None else args.username
            raise SystemExit(f"User '{lookup}' not found in database")
        return user.id
    finally:
        session.close()


def ensure_collection(qdrant_client: QdrantClient, collection_name: str) -> None:
    collections = qdrant_client.get_collections()
    if collection_name not in {col.name for col in collections.collections}:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=3072, distance=qmodels.Distance.COSINE),
        )


def list_supported_files(folder: str) -> Iterable[str]:
    for entry in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, entry)
        if not os.path.isfile(full_path):
            continue
        if get_file_type(entry) == "unknown":
            continue
        yield full_path


def fetch_existing_filenames(
    qdrant_client: QdrantClient, collection_name: str, user_id: int
) -> Set[str]:
    filenames: Set[str] = set()
    offset = None
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id", match=qmodels.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=256,
            offset=offset,
            with_payload=True,
        )
        if not points:
            break
        for point in points:
            payload = point.payload or {}
            filename = payload.get("filename")
            if filename:
                filenames.add(filename)
        if next_offset is None:
            break
        offset = next_offset
    return filenames


def fetch_existing_db_filenames(user_id: int, collection: str) -> Set[str]:
    session: Session = SessionLocal()
    try:
        rows = (
            session.query(PDF.filename)
            .filter(PDF.user_id == user_id, PDF.collection == collection)
            .all()
        )
        return {row[0] for row in rows}
    finally:
        session.close()


def copy_to_temp(path: str) -> str:
    suffix = os.path.splitext(path)[1]
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    shutil.copyfile(path, temp_path)
    return temp_path


async def upload_pdf(
    source_path: str,
    collection: str,
    description: str,
    user_id: int,
) -> None:
    filename = os.path.basename(source_path)
    file_type = get_file_type(filename)
    temp_path = copy_to_temp(source_path)

    session: Session = SessionLocal()
    try:
        pdf_record = PDF(
            user_id=user_id,
            filename=filename,
            description=description,
            status="pending",
            collection=collection,
            file_type=file_type,
        )
        session.add(pdf_record)
        session.commit()
        session.refresh(pdf_record)
    finally:
        session.close()

    await process_file_in_background(
        temp_path=temp_path,
        filename=filename,
        description=description,
        user_id=user_id,
        file_id=pdf_record.id,
        collection=collection,
        file_type=file_type,
    )


async def main() -> None:
    args = parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder '{folder}' does not exist")

    if not config.QDRANT_URL:
        raise SystemExit("QDRANT_URL is not configured")

    user_id = resolve_user_id(args)
    country = os.path.basename(os.path.normpath(folder))

    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    collection_name = f"user_{user_id}_{country}"
    ensure_collection(qdrant_client, collection_name)

    existing_filenames = fetch_existing_filenames(qdrant_client, collection_name, user_id)
    existing_filenames.update(fetch_existing_db_filenames(user_id, country))

    files_to_upload = [path for path in list_supported_files(folder) if os.path.basename(path) not in existing_filenames]

    if not files_to_upload:
        print("No new files to upload.")
        return

    print(f"Found {len(files_to_upload)} new file(s) for collection '{country}'.")
    if args.dry_run:
        for path in files_to_upload:
            print(f"[DRY-RUN] Would upload: {os.path.basename(path)}")
        return

    for path in files_to_upload:
        filename = os.path.basename(path)
        print(f"Uploading {filename} ...", end=" ", flush=True)
        try:
            await upload_pdf(
                source_path=path,
                collection=country,
                description=args.description,
                user_id=user_id,
            )
        except Exception as exc:  # pragma: no cover
            print("failed")
            print(f"  Error: {exc}")
        else:
            print("done")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
