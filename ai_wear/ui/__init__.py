"""UI subpackage.

Importing panels/progress here so ``from . import ui`` (in the addon
__init__) makes ``ui.panels`` / ``ui.progress`` resolvable as attributes of
the package — matching how register()/unregister() address them. Without
this, ``ui.panels.register()`` raises ``AttributeError: module 'ai_wear.ui'
has no attribute 'panels'`` at install time.
"""
from . import panels, progress  # noqa: F401
