import unittest

from src.models import LLMModelFactory, OpenAICompatiblePolicy, VLLMPolicy


class ModelFactoryTests(unittest.TestCase):
    def test_new_provider_profile_config_routes_modules(self):
        factory = LLMModelFactory({
            "default_profile": "solver",
            "providers": {
                "unitprovider": {
                    "adapter": "openai_compatible",
                    "env_prefix": "UNITPROVIDER",
                    "api_key": "unit-key",
                    "base_url": "https://unit.example/v1",
                    "default_model": "unit-default",
                }
            },
            "profiles": {
                "planner_fast": {
                    "provider": "unitprovider",
                    "model": "planner-model",
                    "temperature": 0.2,
                    "max_tokens": 1234,
                },
                "solver": {
                    "provider": "unitprovider",
                    "model": "solver-model",
                    "temperature": 0.5,
                    "max_tokens": 2048,
                },
                "summarizer_long": {
                    "provider": "unitprovider",
                    "model": "summary-model",
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
            },
            "module_profiles": {
                "planner": "planner_fast",
                "summarizer": "summarizer_long",
            },
        })

        planner = factory.describe_module("planner")
        solver = factory.describe_module("solver")
        summarizer = factory.describe_module("summarizer")

        self.assertEqual(planner["profile"], "planner_fast")
        self.assertEqual(planner["model"], "planner-model")
        self.assertEqual(solver["profile"], "solver")
        self.assertEqual(solver["model"], "solver-model")
        self.assertEqual(summarizer["profile"], "summarizer_long")
        self.assertEqual(summarizer["max_tokens"], 4096)

    def test_legacy_backend_config_is_still_supported(self):
        factory = LLMModelFactory({
            "backend": "legacyprovider",
            "temperature": 0.3,
            "max_tokens": 1000,
            "backend_mapping": {
                "planner": "legacyplanner",
            },
            "backend_sampling": {
                "legacyprovider": {"model": "legacy-default"},
                "legacyplanner": {"model": "legacy-planner", "temperature": 0.1},
            },
        })

        default = factory.resolve("default")
        planner = factory.resolve("planner")

        self.assertEqual(default.profile.provider, "legacyprovider")
        self.assertEqual(default.profile.model_name, "legacy-default")
        self.assertEqual(planner.profile.provider, "legacyplanner")
        self.assertEqual(planner.profile.model_name, "legacy-planner")
        self.assertEqual(planner.profile.temperature, 0.1)

    def test_missing_api_key_error_is_explicit(self):
        factory = LLMModelFactory({
            "providers": {
                "missingkey": {
                    "adapter": "openai_compatible",
                    "env_prefix": "NO_SUCH_PROVIDER_FOR_TESTS",
                    "base_url": "https://missing.example/v1",
                    "default_model": "missing-model",
                }
            },
            "profiles": {
                "default": {"provider": "missingkey"},
            },
        })

        with self.assertRaisesRegex(ValueError, "NO_SUCH_PROVIDER_FOR_TESTS_API_KEY"):
            factory.resolve("default").to_policy_kwargs()

    def test_cache_key_excludes_api_key(self):
        factory = LLMModelFactory({
            "providers": {
                "secure": {
                    "adapter": "openai_compatible",
                    "api_key": "secret-key-that-must-not-leak",
                    "base_url": "https://secure.example/v1",
                    "default_model": "secure-model",
                }
            },
            "profiles": {
                "default": {"provider": "secure"},
            },
        })
        resolved = factory.resolve("default")
        kwargs = resolved.to_policy_kwargs()

        cache_key = factory._cache_key(resolved, kwargs)

        self.assertNotIn("secret-key-that-must-not-leak", cache_key)
        self.assertNotIn("api_key", cache_key)
        self.assertIn("secure-model", cache_key)

    def test_vllm_policy_name_remains_backward_compatible(self):
        self.assertIs(VLLMPolicy, OpenAICompatiblePolicy)


if __name__ == "__main__":
    unittest.main()
