from __future__ import annotations
import datetime as dt
import logging
from typing import Iterable, List

try:
    from sqlalchemy import (create_engine, Column, Integer, String, Text,
                            Float, Boolean, DateTime)
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError:
    # If SQLAlchemy isn't installed the Database class will not work.
    create_engine = None  # type: ignore
    Column = Integer = String = Text = Float = Boolean = DateTime = None  # type: ignore
    declarative_base = None  # type: ignore
    sessionmaker = None  # type: ignore

from structures import Post

logger = logging.getLogger(__name__)

Base = declarative_base() if declarative_base is not None else None

class PostModel(Base):  # type: ignore
    """
    SQLAlchemy model representing a social media post.  Fields mirror the
    attributes of the Post dataclass but are persisted in a relational
    database.
    """

    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    platform = Column(String, nullable=False)
    keyword = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    url = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    author = Column(String, nullable=True)
    predicted_label = Column(String, nullable=True)
    prediction_score = Column(Float, nullable=True)
    fact_check_url = Column(String, nullable=True)
    fact_check_rating = Column(String, nullable=True)
    fact_check_publisher = Column(String, nullable=True)
    inserted_at = Column(DateTime, nullable=False, default=dt.datetime.utcnow)


class Database:
    """
    Handle persistence of posts using SQLAlchemy.

    Supports both SQLite (default) and PostgreSQL (or any other database
    supported by SQLAlchemy).  The database URL should be provided in the
    form accepted by SQLAlchemy, e.g. `sqlite:///data.db` or
    `postgresql+psycopg2://user:password@localhost/dbname`.
    """

    def __init__(self, db_url: str = "sqlite:///data.db") -> None:
        if create_engine is None:
            raise ImportError("SQLAlchemy is required for Database but it's not installed.")
        self.engine = create_engine(db_url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def insert_posts(self, posts: Iterable[Post]) -> None:
        """
        Insert a list of Post objects into the database.  Existing rows with
        the same URL will be ignored to avoid duplicates.
        """
        session = self.Session()
        try:
            for post in posts:
                # Check for duplicates
                exists = session.query(PostModel).filter_by(url=post.url).first()
                if exists:
                    exists.predicted_label = post.predicted_label
                    exists.prediction_score = post.prediction_score
                    exists.fact_check_url = post.fact_check_url
                    exists.fact_check_rating = post.fact_check_rating
                    exists.fact_check_publisher = post.fact_check_publisher
                    continue
                model = PostModel(
                    platform=post.platform,
                    keyword=post.keyword,
                    content=post.content,
                    url=post.url,
                    created_at=post.created_at,
                    author=post.author,
                    predicted_label=post.predicted_label,
                    prediction_score=post.prediction_score,
                    fact_check_url=post.fact_check_url,
                    fact_check_rating=post.fact_check_rating,
                    fact_check_publisher=post.fact_check_publisher,
                    inserted_at=dt.datetime.utcnow(),
                )
                session.add(model)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to insert posts into database")
        finally:
            session.close()

    def get_posts(self, limit: int = 1000) -> List[PostModel]:
        """
        Return the most recent posts from the database up to the specified limit.
        """
        session = self.Session()
        try:
            posts = session.query(PostModel).order_by(PostModel.inserted_at.desc()).limit(limit).all()
            return posts
        finally:
            session.close()
