"""
Environment setup script for KERAG with multiple LLM providers.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_env_file():
    """Create a .env file template for API keys."""

    env_content = """# KERAG Environment Configuration
# Copy this file to .env and fill in your API keys

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Configuration
GOOGLE_API_KEY=your_google_api_key_here

# xAI Configuration
XAI_API_KEY=your_xai_api_key_here

# KERAG Configuration
KERAG_LLM_MODEL=gpt-4
KERAG_DEBUG=false
KERAG_TIMEOUT=30

# Knowledge Base Configuration
SPARQL_ENDPOINT=http://dbpedia.org/sparql
API_BASE_URL=https://api.example-knowledge-base.com
API_KEY=your_knowledge_base_api_key_here

# Performance Configuration
KERAG_MAX_HOPS=3
KERAG_MAX_ENTITIES=1000
KERAG_RELEVANCE_THRESHOLD=0.5
KERAG_BATCH_SIZE=10
"""

    env_file = Path(".env")

    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != "y":
            print("Keeping existing .env file")
            return

    with open(env_file, "w") as f:
        f.write(env_content)

    print("✅ Created .env file template")
    print("📝 Please edit .env and add your API keys")


def check_api_keys():
    """Check which API keys are available."""

    api_keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "xAI": "XAI_API_KEY",
    }

    print("🔑 API Key Status:")
    print("-" * 30)

    available_keys = []

    for provider, key_name in api_keys.items():
        key_value = os.getenv(key_name)
        if key_value and key_value != f"your_{key_name.lower()}_here":
            print(f"✅ {provider}: {key_name} is set")
            available_keys.append(provider)
        else:
            print(f"❌ {provider}: {key_name} not found")

    if available_keys:
        print(f"\n🎉 You can use: {', '.join(available_keys)}")
    else:
        print(
            "\n⚠️  No API keys found. Please set them in your .env file or environment."
        )

    return available_keys


def get_api_key_instructions():
    """Print instructions for getting API keys."""

    instructions = """
🔑 How to get API keys:

1. OpenAI (GPT models):
   - Visit: https://platform.openai.com/api-keys
   - Create account and generate API key
   - Set: export OPENAI_API_KEY=your_key_here

2. Anthropic (Claude models):
   - Visit: https://console.anthropic.com/
   - Create account and generate API key
   - Set: export ANTHROPIC_API_KEY=your_key_here

3. Google (Gemini models):
   - Visit: https://makersuite.google.com/app/apikey
   - Create account and generate API key
   - Set: export GOOGLE_API_KEY=your_key_here

4. xAI (Grok models):
   - Visit: https://console.x.ai/
   - Create account and generate API key
   - Set: export XAI_API_KEY=your_key_here

📝 You can also create a .env file with these keys:
   python setup_env.py --create-env
"""

    print(instructions)


def main():
    """Main setup function."""

    import argparse

    parser = argparse.ArgumentParser(description="KERAG Environment Setup")
    parser.add_argument(
        "--create-env", action="store_true", help="Create .env template file"
    )
    parser.add_argument(
        "--check-keys", action="store_true", help="Check available API keys"
    )
    parser.add_argument(
        "--instructions", action="store_true", help="Show API key instructions"
    )

    args = parser.parse_args()

    if args.create_env:
        create_env_file()
    elif args.check_keys:
        check_api_keys()
    elif args.instructions:
        get_api_key_instructions()
    else:
        # Default behavior - show all
        print("🚀 KERAG Multi-LLM Setup")
        print("=" * 40)

        # Check current keys
        available_keys = check_api_keys()

        if not available_keys:
            print("\n" + "=" * 40)
            get_api_key_instructions()

            # Offer to create .env file
            response = input("\nWould you like to create a .env template file? (y/N): ")
            if response.lower() == "y":
                create_env_file()

        print(f"\n🎯 Next steps:")
        print(f"   1. Set your API keys (see instructions above)")
        print(f"   2. Install dependencies: pip install -r requirements.txt")
        print(f"   3. Run examples: python examples/multi_llm_example.py")


if __name__ == "__main__":
    main()
