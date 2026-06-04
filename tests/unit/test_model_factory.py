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

    def test_response_format_is_forwarded_to_policy_kwargs(self):
        factory = LLMModelFactory({
            "providers": {
                "unitprovider": {
                    "adapter": "openai_compatible",
                    "api_key": "unit-key",
                    "base_url": "https://unit.example/v1",
                    "default_model": "unit-default",
                }
            },
            "profiles": {
                "default": {
                    "provider": "unitprovider",
                    "response_format": {"type": "json_object"},
                },
            },
        })

        kwargs = factory.resolve("default").to_policy_kwargs()

        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_vllm_policy_name_remains_backward_compatible(self):
        self.assertIs(VLLMPolicy, OpenAICompatiblePolicy)

    def test_active_preset_overrides_module_profiles(self):
        factory = LLMModelFactory({
            "active_preset": "low_cost",
            "default_profile": "solver",
            "providers": {
                "unitprovider": {
                    "adapter": "openai_compatible",
                    "api_key": "unit-key",
                    "base_url": "https://unit.example/v1",
                    "default_model": "unit-default",
                }
            },
            "profiles": {
                "planner": {"provider": "unitprovider", "model": "planner-model"},
                "solver": {"provider": "unitprovider", "model": "solver-model"},
                "summarizer": {"provider": "unitprovider", "model": "summary-model"},
                "compressor": {"provider": "unitprovider", "model": "compressor-model"},
            },
            "module_profiles": {
                "summarizer": "summarizer",
            },
            "presets": {
                "balanced": {
                    "module_profiles": {"summarizer": "summarizer"},
                },
                "low_cost": {
                    "default_profile": "solver",
                    "module_profiles": {
                        "planner": "planner",
                        "solver": "solver",
                        "summarizer": "compressor",
                        "compressor": "compressor",
                    },
                },
            },
        })

        self.assertEqual(factory.active_preset, "low_cost")
        self.assertEqual(factory.describe_module("summarizer")["profile"], "compressor")
        self.assertEqual(factory.describe_module("summarizer")["active_preset"], "low_cost")
        self.assertEqual(factory.describe_module("summarizer")["model"], "compressor-model")

    def test_unknown_active_preset_errors(self):
        with self.assertRaisesRegex(ValueError, "Unknown model preset"):
            LLMModelFactory({
                "active_preset": "does_not_exist",
                "presets": {"balanced": {}},
            })

    def test_preset_missing_profile_errors(self):
        with self.assertRaisesRegex(ValueError, "missing profile"):
            LLMModelFactory({
                "active_preset": "broken",
                "providers": {
                    "unitprovider": {
                        "adapter": "openai_compatible",
                        "api_key": "unit-key",
                        "base_url": "https://unit.example/v1",
                        "default_model": "unit-default",
                    }
                },
                "profiles": {
                    "default": {"provider": "unitprovider"},
                    "solver": {"provider": "unitprovider"},
                },
                "presets": {
                    "broken": {
                        "module_profiles": {"planner": "missing_planner_profile"},
                    },
                },
            })

    def test_describe_module_records_preset_but_not_api_key(self):
        factory = LLMModelFactory({
            "active_preset": "balanced",
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
                "solver": {"provider": "secure", "model": "solver-model"},
            },
            "presets": {
                "balanced": {
                    "default_profile": "solver",
                    "module_profiles": {"solver": "solver"},
                },
            },
        })

        description = str(factory.describe_module("solver"))

        self.assertIn("balanced", description)
        self.assertNotIn("secret-key-that-must-not-leak", description)
        self.assertNotIn("api_key", description)

    def test_known_openai_compatible_provider_defaults_are_available(self):
        for provider in ["openai", "openrouter", "siliconflow", "moonshot", "kimi", "dashscope", "qwen", "deepseek"]:
            factory = LLMModelFactory({
                "providers": {
                    provider: {
                        "allow_missing_api_key": True,
                    }
                },
                "profiles": {
                    "default": {"provider": provider},
                },
            })
            resolved = factory.resolve("default")

            self.assertEqual(resolved.provider.adapter, "openai_compatible")
            self.assertTrue(resolved.provider.base_url)
            self.assertTrue(resolved.profile.model_name or resolved.provider.default_model)


if __name__ == "__main__":
    unittest.main()
