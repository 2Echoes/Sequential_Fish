import sys
from Sequential_Fish import viewer, pipeline

def main():

    MODULES = ['viewer', 'pipeline']

    if len(sys.argv) < 2:
        print("Usage: python -m my_package <module> [args...]")
        print("Available modules: {0}".format(MODULES))
        sys.exit(1)

    module = sys.argv[1]
    args = sys.argv[2:]

    if module == "viewer":
        viewer.run(args)
    elif module == "pipeline":
        pipeline.run(args)
    else:
        print(f"Unknown module: {module}")
        print("Available modules: {0}".format(MODULES))
        sys.exit(1)

if __name__ == "__main__":
    main()
