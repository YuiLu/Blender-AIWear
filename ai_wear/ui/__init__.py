"""UI subpackage.

Importing panels here so ``from . import ui`` (in the addon ``__init__``)
makes ``ui.panels`` resolvable as an attribute of the package. Without
this, ``ui.panels.register()`` raises ``AttributeError: module 'ai_wear.ui'
has no attribute 'panels'`` at install time.
"""
from . import panels  # noqa: F401
