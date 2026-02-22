from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from rest_framework.test import APITestCase

from signals.feature_flags import FEATURE_FLAGS_VERSION
from signals.models import StrategyConfig


@override_settings(
    MIDDLEWARE=[
        "django.middleware.security.SecurityMiddleware",
        "django.contrib.sessions.middleware.SessionMiddleware",
        "django.middleware.common.CommonMiddleware",
        "django.middleware.csrf.CsrfViewMiddleware",
        "django.contrib.auth.middleware.AuthenticationMiddleware",
        "django.contrib.messages.middleware.MessageMiddleware",
        "django.middleware.clickjacking.XFrameOptionsMiddleware",
    ]
)
class StrategyFeatureFlagsApiTest(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="tester",
            email="tester@example.com",
            password="secret-123",
        )
        self.client.force_authenticate(self.user)

    def test_set_and_toggle_feature_flag(self):
        set_url = reverse("strategy-config-set-feature")
        toggle_url = reverse("strategy-config-toggle-feature")
        features_url = reverse("strategy-config-features")

        resp_set = self.client.post(
            set_url,
            {"name": "feature_mod_meanrev", "enabled": False},
            format="json",
        )
        self.assertEqual(resp_set.status_code, 200)
        self.assertFalse(resp_set.data["enabled"])

        row = StrategyConfig.objects.get(
            name="feature_mod_meanrev",
            version=FEATURE_FLAGS_VERSION,
        )
        self.assertFalse(row.enabled)

        resp_toggle = self.client.post(
            toggle_url,
            {"name": "feature_mod_meanrev"},
            format="json",
        )
        self.assertEqual(resp_toggle.status_code, 200)
        self.assertTrue(resp_toggle.data["enabled"])

        row.refresh_from_db()
        self.assertTrue(row.enabled)

        resp_features = self.client.get(features_url)
        self.assertEqual(resp_features.status_code, 200)
        resolved = resp_features.data.get("resolved", {})
        self.assertIn("feature_mod_meanrev", resolved)
        self.assertTrue(resolved["feature_mod_meanrev"])
