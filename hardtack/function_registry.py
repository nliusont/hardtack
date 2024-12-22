# function_registry.py

# The FUNCTION_REGISTRY is intentionally left empty.
# This is because we want to avoid circular imports. Instead of importing functions directly into this file,
# we will populate the registry with function references dynamically in agent.py after all modules are loaded.
# This avoids the circular dependency that would occur if we imported functions from agent.py directly here.

FUNCTION_REGISTRY = {}
