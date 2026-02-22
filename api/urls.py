from rest_framework import routers

from django.urls import include, path

from .views import (
    InstrumentViewSet,
    OrderViewSet,
    PositionViewSet,
    RiskEventViewSet,
    SignalViewSet,
    StrategyConfigViewSet,
)

router = routers.DefaultRouter()
router.register(r"instruments", InstrumentViewSet)
router.register(r"signals", SignalViewSet, basename="signal")
router.register(r"positions", PositionViewSet, basename="position")
router.register(r"orders", OrderViewSet, basename="order")
router.register(r"risk", RiskEventViewSet, basename="risk")
router.register(r"config/strategy", StrategyConfigViewSet, basename="strategy-config")

urlpatterns = [
    path("", include(router.urls)),
]
