from django.test import SimpleTestCase

from core.toon_validator import validate_toon_text


class ToonValidatorTest(SimpleTestCase):
    def test_valid_toon_minimal(self):
        text = """
# SAMPLE
FORMAT: TOON
MODE: TOKEN_OPTIMIZED
SECTION: HARD_CONSTRAINTS
NO:
- LEVERAGE increase
ALWAYS:
- WALK_FORWARD_VALIDATION
END_OF_TOON_CONTEXT
"""
        res = validate_toon_text(text, path="x.toon.md")
        self.assertTrue(res.valid)
        self.assertEqual(res.errors, [])

    def test_invalid_missing_markers(self):
        text = "SECTION: HARD_CONSTRAINTS\nNO:\n- x\n"
        res = validate_toon_text(text, path="bad.toon.md")
        self.assertFalse(res.valid)
        self.assertTrue(any("FORMAT: TOON" in e for e in res.errors))
        self.assertTrue(any("END_OF_TOON_CONTEXT" in e for e in res.errors))

    def test_invalid_long_narrative_block(self):
        text = """
FORMAT: TOON
SECTION: HARD_CONSTRAINTS
this is line one
this is line two
this is line three
this is line four
END_OF_TOON_CONTEXT
"""
        res = validate_toon_text(text, path="narrative.toon.md")
        self.assertFalse(res.valid)
        self.assertTrue(any("narrative block too long" in e for e in res.errors))
