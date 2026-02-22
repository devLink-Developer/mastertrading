from rest_framework import serializers

from core.models import Instrument
from execution.models import Order, Position
from risk.models import RiskEvent
from signals.models import Signal, StrategyConfig


class InstrumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Instrument
        fields = "__all__"


class SignalSerializer(serializers.ModelSerializer):
    instrument = InstrumentSerializer(read_only=True)

    class Meta:
        model = Signal
        fields = "__all__"


class PositionSerializer(serializers.ModelSerializer):
    instrument = InstrumentSerializer(read_only=True)

    class Meta:
        model = Position
        fields = "__all__"


class OrderSerializer(serializers.ModelSerializer):
    instrument = InstrumentSerializer(read_only=True)

    class Meta:
        model = Order
        fields = "__all__"


class RiskEventSerializer(serializers.ModelSerializer):
    instrument = InstrumentSerializer(read_only=True)

    class Meta:
        model = RiskEvent
        fields = "__all__"


class StrategyConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = StrategyConfig
        fields = "__all__"
