# views.py

from django.http import JsonResponse
from django.db import connection


def health_check(request):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version();")
            db_version = cursor.fetchone()[0]

            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            vector_version = cursor.fetchone()

        return JsonResponse({
            "backend": "connected",
            "database": "connected",
            "postgres_version": db_version,
            "pgvector": vector_version[0] if vector_version else None,
        })

    except Exception as e:
        return JsonResponse({
            "backend": "connected",
            "database": "error",
            "error": str(e),
        }, status=500)