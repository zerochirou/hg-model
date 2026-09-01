import os
from dotenv import load_dotenv
from supabase import create_client, Client


def load_road_example():
    """
    Mengambil data riwayat deteksi sensor dari tabel pothole_detections di Supabase.
    Memiliki fallback aman jika koneksi/kebijakan RLS Supabase gagal.
    """
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("[Warning] SUPABASE_URL atau SUPABASE_KEY tidak ditemukan di .env")
        return []

    try:
        supabase = create_client(url, key)
        print("Supabase connected")

        response = (
            supabase
            .table("pothole_detections")
            .select("*")
            .execute()
        )

        data = response.data
        return data
    except Exception as e:
        print(f"[Notice] Gagal mengambil data dari Supabase ({e}). Menggunakan data lokal atau list kosong.")
        return []


# Alias untuk kompatibilitas
pull_road_data = load_road_example