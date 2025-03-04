import sys
from Sequential_Fish import viewer, pipeline
from Sequential_Fish.pipeline.runner import launch_script, script_folder

def main():

    MODULES = ['viewer', 'pipeline']
    pipeline_scripts = ['input', 'detection', 'segmentation', 'drift', 'alignement', 'washout', 'quantification']

    if len(sys.argv) < 2:
        print("Usage: python -m my_package <module> [args...]")
        print("Available modules: {0}".format(MODULES))
        sys.exit(1)

    module = sys.argv[1]
    submodules = sys.argv[2:]

    if module == "viewer":

        if len(submodules) > 0 :
            print(f"No argument for viewer. Ignoring passed arguments : {submodules}")

        viewer.run()
        
    elif module == "pipeline":
        if len(submodules) == 0 :
            pipeline.run()
        else :
            if not all([script in pipeline_scripts for script in submodules]) :
                print(f"Unknown pipeline scripts. \nChoose from : {pipeline_scripts}")
            else :
                for script in submodules : 
                    if not script.endswith('.py') : script += ".py"
                    launch_script(script_folder + '/' + script)
    else:
        print(f"Unknown module: {module}")
        print("Available modules: {0}".format(MODULES))
        sys.exit(1)

if __name__ == "__main__":
    main()
