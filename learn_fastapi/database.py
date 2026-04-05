from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQL_ALCHAMY_DATABASE_URL = 'sqlite:///learn_fastapi/db/sql_db.db'
CONNECT_ARGS = {"check_same_thread": False}

engine = create_engine(SQL_ALCHAMY_DATABASE_URL, connect_args=CONNECT_ARGS)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()
