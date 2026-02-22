from django.shortcuts import get_object_or_404
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from core.models import Instrument
from execution.models import Order, Position
from risk.models import RiskEvent
from signals.feature_flags import (
    FEATURE_FLAGS_VERSION,
    feature_flag_defaults,
    resolve_runtime_flags,
)
from signals.models import Signal, StrategyConfig
from .serializers import (
    InstrumentSerializer,
    OrderSerializer,
    PositionSerializer,
    RiskEventSerializer,
    SignalSerializer,
    StrategyConfigSerializer,
)


class InstrumentViewSet(viewsets.ModelViewSet):
    queryset = Instrument.objects.all()
    serializer_class = InstrumentSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @action(detail=True, methods=["post"])
    def enable(self, request, pk=None):
        instrument = self.get_object()
        instrument.enabled = True
        instrument.save(update_fields=["enabled"])
        return Response({"status": "enabled"})

    @action(detail=True, methods=["post"])
    def disable(self, request, pk=None):
        instrument = self.get_object()
        instrument.enabled = False
        instrument.save(update_fields=["enabled"])
        return Response({"status": "disabled"})


class SignalViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Signal.objects.select_related("instrument").all()
    serializer_class = SignalSerializer
    permission_classes = [permissions.AllowAny]


class PositionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Position.objects.select_related("instrument").all()
    serializer_class = PositionSerializer
    permission_classes = [permissions.AllowAny]


class OrderViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Order.objects.select_related("instrument").all()
    serializer_class = OrderSerializer
    permission_classes = [permissions.AllowAny]


class RiskEventViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = RiskEvent.objects.select_related("instrument").all()
    serializer_class = RiskEventSerializer
    permission_classes = [permissions.AllowAny]


def _parse_enabled_value(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    txt = str(raw).strip().lower()
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    raise ValueError("enabled must be a boolean")


class StrategyConfigViewSet(viewsets.ModelViewSet):
    queryset = StrategyConfig.objects.all().order_by("name", "version")
    serializer_class = StrategyConfigSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @action(detail=True, methods=["post"])
    def toggle(self, request, pk=None):
        strategy = get_object_or_404(StrategyConfig, pk=pk)
        strategy.enabled = not strategy.enabled
        strategy.save(update_fields=["enabled"])
        return Response({"enabled": strategy.enabled}, status=status.HTTP_200_OK)

    @action(detail=False, methods=["get"])
    def features(self, request):
        defaults = feature_flag_defaults()
        resolved = resolve_runtime_flags()
        rows = (
            StrategyConfig.objects.filter(
                version=FEATURE_FLAGS_VERSION,
                name__in=list(defaults.keys()),
            )
            .order_by("name")
            .values("id", "name", "enabled", "version", "created_at")
        )
        return Response(
            {
                "version": FEATURE_FLAGS_VERSION,
                "defaults": defaults,
                "resolved": resolved,
                "rows": list(rows),
            },
            status=status.HTTP_200_OK,
        )

    @action(detail=False, methods=["post"])
    def set_feature(self, request):
        name = str(request.data.get("name", "")).strip()
        if not name:
            return Response(
                {"detail": "name is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            enabled = _parse_enabled_value(request.data.get("enabled"))
        except ValueError as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        row, _ = StrategyConfig.objects.update_or_create(
            name=name,
            version=FEATURE_FLAGS_VERSION,
            defaults={"enabled": enabled, "params_json": {"feature_flag": True}},
        )
        return Response(
            {
                "id": row.id,
                "name": row.name,
                "version": row.version,
                "enabled": row.enabled,
            },
            status=status.HTTP_200_OK,
        )

    @action(detail=False, methods=["post"])
    def toggle_feature(self, request):
        name = str(request.data.get("name", "")).strip()
        if not name:
            return Response(
                {"detail": "name is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        defaults = feature_flag_defaults()
        default_enabled = bool(defaults.get(name, True))
        row, _ = StrategyConfig.objects.get_or_create(
            name=name,
            version=FEATURE_FLAGS_VERSION,
            defaults={"enabled": default_enabled, "params_json": {"feature_flag": True}},
        )
        row.enabled = not bool(row.enabled)
        row.save(update_fields=["enabled"])
        return Response(
            {
                "id": row.id,
                "name": row.name,
                "version": row.version,
                "enabled": row.enabled,
            },
            status=status.HTTP_200_OK,
        )
