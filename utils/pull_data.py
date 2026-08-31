import os
from dotenv import load_dotenv
from supabase import create_client, Client


def load_road_example():
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

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