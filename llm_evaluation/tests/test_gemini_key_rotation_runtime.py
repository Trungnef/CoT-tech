import sys
import importlib
import types as _types
import unittest
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Prepare a fake google.generativeai module to simulate quota error on first key
fake_genai = _types.SimpleNamespace()
fake_genai._counter = 0

class FakeModel:
    def __init__(self, model_name=None):
        self.model_name = model_name
    def generate_content(self, prompt, generation_config=None):
        # First call -> raise quota error, subsequent -> return success
        fake_genai._counter += 1
        if fake_genai._counter == 1:
            e = Exception("Quota exceeded")
            # Attach status_code to mimic HTTP-like exception
            setattr(e, 'status_code', 429)
            raise e
        return _types.SimpleNamespace(text=f"ok-called-{fake_genai._counter}")

fake_genai.GenerativeModel = lambda model_name=None: FakeModel(model_name=model_name)
# Provide a configure function expected by model_interface._get_gemini_client
def _fake_configure(api_key=None):
    setattr(fake_genai, 'last_api_key', api_key)

fake_genai.configure = _fake_configure

# Inject fake module BEFORE importing model_interface so it gets bound
_google_mod = _types.ModuleType('google')
sys.modules['google'] = _google_mod
sys.modules['google.generativeai'] = fake_genai
setattr(sys.modules['google'], 'generativeai', fake_genai)

# Also inject lightweight mocks for heavy dependencies to avoid import-time failures
sys.modules['torch'] = _types.SimpleNamespace(cuda=_types.SimpleNamespace(is_available=lambda : False), __version__='1.0')
sys.modules['transformers'] = _types.SimpleNamespace()
sys.modules['huggingface_hub'] = _types.SimpleNamespace()

# Minimal tenacity shim used only for import-time (decorators become no-ops)
def _noop_decorator(*a, **k):
    def _inner(f):
        return f
    return _inner

sys.modules['tenacity'] = _types.SimpleNamespace(
    retry=_noop_decorator,
    stop_after_attempt=lambda n: None,
    wait_exponential=lambda *a, **k: None,
    retry_if_exception_type=lambda *a, **k: None,
    wait_fixed=lambda *a, **k: None,
    retry_if_result=lambda *a, **k: None,
    retry_if_exception=lambda *a, **k: None
)
# Provide minimal transformer symbols used at import time
class _Dummy:
    def __init__(self, *a, **k):
        pass

_trans = _types.SimpleNamespace(
    AutoTokenizer=_Dummy,
    AutoModelForCausalLM=_Dummy,
    BitsAndBytesConfig=_Dummy
)
sys.modules['transformers'] = _trans
sys.modules['dotenv'] = _types.SimpleNamespace(load_dotenv=lambda : None)

# Import the module under test without importing the whole package (avoid heavy deps)
import importlib.util
from pathlib import Path

module_path = Path(__file__).parent.parent / "core" / "model_interface.py"
spec = importlib.util.spec_from_file_location("model_interface_for_test", str(module_path))
model_interface = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model_interface
spec.loader.exec_module(model_interface)

class TestGeminiKeyRotationRuntime(unittest.TestCase):
    def test_key_rotation_on_quota(self):
        # Set two keys
        model_interface.config.GEMINI_API_KEYS = ['key1', 'key2']

        interface = model_interface.ModelInterface()

        # Ensure starting index is 0
        self.assertEqual(interface.current_gemini_key_index, 0)

        # First call should raise (quota on key1) and rotate to next key
        with self.assertRaises(Exception):
            interface._generate_with_gemini_impl("prompt", {"max_tokens": 10})

        # After handling quota, current key index should have advanced to 1
        self.assertEqual(interface.current_gemini_key_index, 1)

        # Second call should succeed using key2
        response, stats = interface._generate_with_gemini_impl("prompt", {"max_tokens": 10})
        self.assertEqual(response, "ok-called-2")
        self.assertFalse(stats.get('has_error', False))

if __name__ == '__main__':
    unittest.main()
