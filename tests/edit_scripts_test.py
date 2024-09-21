"""Unit tests for edit_scripts."""

import unittest

from udtube.data import edit_scripts


class EScriptTest(unittest.TestCase):
    def assertValid(
        self, istring: str, ostring: str, script: edit_scripts.EditScript
    ) -> None:
        # Asserts the edit script produces the expected result.
        exp_ostring = script.apply(istring)
        self.assertEqual(ostring, exp_ostring)

    def testEmpty(self):
        token = ""
        script = edit_scripts.EditScript(token, token)
        self.assertEqual(str(script), "")
        self.assertValid(token, token, script)
        rscript = edit_scripts.ReverseEditScript(token, token)
        self.assertEqual(str(rscript), "")
        self.assertValid(token, token, rscript)

    def testIdentity(self):
        token = "identify"
        script = edit_scripts.EditScript(token, token)
        self.assertEqual(str(script), "")
        self.assertValid(token, token, script)
        rscript = edit_scripts.ReverseEditScript(token, token)
        self.assertEqual(str(rscript), "")
        self.assertValid(token, token, rscript)
        # Works despite length mismatch.
        token = "idempotent"
        self.assertValid(token, token, script)
        self.assertValid(token, token, rscript)

    def testPrefixation(self):
        before = "umibig"
        after = "ibig"
        script = edit_scripts.EditScript(before, after)
        self.assertValid(before, after, script)

    def testInfixation(self):
        before = "lipad"
        after = "lumipad"
        script = edit_scripts.EditScript(before, after)
        self.assertValid(before, after, script)

    def testStemChange(self):
        before = "Väter"
        after = "Vater"
        script = edit_scripts.EditScript(before, after)
        self.assertValid(before, after, script)
        rscript = edit_scripts.ReverseEditScript(before, after)
        self.assertValid(before, after, rscript)

    def testStemChangeWithSuffixation(self):
        before = "Bäume"
        after = "Baum"
        rscript = edit_scripts.ReverseEditScript(before, after)
        self.assertValid(before, after, rscript)

    def testSuffixation(self):
        before = "столов"
        after = "стол"
        rscript = edit_scripts.ReverseEditScript(before, after)
        self.assertValid(before, after, rscript)
        # Works despite length mismatch.
        self.assertValid("градусов", "градус", rscript)

    def testSpanishExample(self):
        # After Chrupała et al. 2008, p. 2362.
        before = "pidieron"
        after = "pedir"
        rscript = edit_scripts.ReverseEditScript(before, after)
        self.assertEqual(str(rscript), "~|~||~|||~e")
        self.assertValid(before, after, rscript)
        # Also testing roundtrip serialization/deserialization.
        rscript = edit_scripts.ReverseEditScript.fromtag(str(rscript))
        # Works despite length mismatch.
        self.assertValid("repitieron", "repetir", rscript)


if __name__ == "__main__":
    unittest.main()
