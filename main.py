import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Generator, Literal

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    DateTime,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

# 1. Membaca konfigurasi dari file .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if not DATABASE_URL or not SECRET_KEY:
    raise RuntimeError(
        "DATABASE_URL dan SECRET_KEY harus diatur di file .env"
    )

# 2. Membuat koneksi ke PostgreSQL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# 3. Dasar untuk model database
class Base(DeclarativeBase):
    pass


# 4. Struktur tabel users
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )
    name: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default="mahasiswa"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


# 5. Membuat tabel ketika aplikasi dijalankan
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Statistics Story AI Backend",
    version="0.1.0",
    lifespan=lifespan,
)

# 6. Mengizinkan frontend mengakses backend
origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 7. Aturan validasi data masuk
class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    role: Literal["mahasiswa", "dosen", "peneliti"] | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


# 8. Membuka dan menutup sesi database
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 9. Menyimpan password sebagai hash
def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=422,
            detail="Password terlalu panjang untuk bcrypt",
        )

    hashed = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    )
    return hashed.decode("utf-8")


# 10. Memeriksa password ketika login
def verify_password(password: str, password_hash: str) -> bool:
    if len(password.encode("utf-8")) > 72:
        return False

    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except ValueError:
        return False


# 11. Membuat cookie sesi setelah login
def create_session_cookie(response: Response, user: User) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    token = jwt.encode(
        {
            "sub": str(user.id),
            "exp": expires_at,
        },
        SECRET_KEY,
        algorithm="HS256",
    )

    production = ENVIRONMENT == "production"

    response.set_cookie(
        key="statistics_session",
        value=token,
        httponly=True,
        secure=production,
        samesite="none" if production else "lax",
        max_age=7 * 24 * 60 * 60,
        path="/",
    )


# 12. Memeriksa apakah backend hidup
@app.get("/")
def root():
    return {"message": "Statistics Story AI backend berjalan"}


# 13. Endpoint register
@app.post("/auth/register", status_code=201)
def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
):
    name = payload.name.strip()
    email = str(payload.email).strip().lower()

    if len(name) < 2:
        raise HTTPException(
            status_code=422,
            detail="Nama minimal 2 karakter",
        )

    existing_user = db.scalar(
        select(User).where(User.email == email)
    )

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Email sudah terdaftar",
        )

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(payload.password),
        role=payload.role or "mahasiswa",
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Email sudah terdaftar",
        )

    return {
        "message": "Registrasi berhasil",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
        },
    }


# 14. Endpoint login
@app.post("/auth/login")
def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    email = str(payload.email).strip().lower()

    user = db.scalar(
        select(User).where(User.email == email)
    )

    if not user or not verify_password(
        payload.password, user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Email atau password salah",
        )

    create_session_cookie(response, user)

    return {
        "message": "Login berhasil",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
        },
    }


# 15. Memeriksa sesi pengguna
@app.get("/auth/me")
def current_user(
    request: Request,
    db: Session = Depends(get_db),
):
    token = request.cookies.get("statistics_session")

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Belum login",
        )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=["HS256"],
        )
        user_id = int(payload["sub"])
    except (
        jwt.InvalidTokenError,
        KeyError,
        TypeError,
        ValueError,
    ):
        raise HTTPException(
            status_code=401,
            detail="Sesi tidak valid atau kedaluwarsa",
        )

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Pengguna tidak ditemukan",
        )

    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
    }