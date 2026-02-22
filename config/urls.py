from django.contrib import admin
from django.http import JsonResponse, HttpResponse
from django.urls import include, path

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


def health_view(request):
    return JsonResponse({"status": "ok"})


def metrics_view(request):
    return HttpResponse(generate_latest(), content_type=CONTENT_TYPE_LATEST)


urlpatterns = [
    path("admin/", admin.site.urls),
    path("health", health_view, name="health"),
    path("metrics", metrics_view, name="metrics"),
    path("", include("api.urls")),
]
