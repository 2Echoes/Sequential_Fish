"""
Entry point to launch Sequential Fish analysis pipeline.
"""

def run(*args) :
    from .runner import main
    main()