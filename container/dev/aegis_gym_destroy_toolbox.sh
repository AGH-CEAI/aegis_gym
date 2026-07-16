#!/usr/bin/env bash

workspace_name="aegis_gym-dev"
image_name="localhost/aegis-gym-dev:devel"

while getopts ":w:i:" option; do
    case "$option" in
        w)
            workspace_name="$OPTARG"
            ;;
        i)
            image_name="$OPTARG"
            ;;
        *)
            echo "Usage: $0 -w workspace_name [-i image_name]"
            exit 1
            ;;
esac
done

if [[ -z "$workspace_name" ]]; then
    echo "Error: workspace name is required."
    echo "Usage: $0 -w workspace_name [-i image_name]"
    exit 1
fi

echo "Toolbox name: $workspace_name"
echo "Image name: $image_name"

while true; do
    read -r -p "Do you want to stop and remove the Toolbox and delete the image? [y/N] " choice

    case "$choice" in
        y|Y)
            echo "Stopping Toolbox container..."

            if podman container exists "$workspace_name"; then
                podman stop "$workspace_name"
            else
                echo "Container '$workspace_name' does not exist."
            fi

            echo "Removing Toolbox..."

            if toolbox list --containers | grep -q "$workspace_name"; then
                toolbox rm --force "$workspace_name"
            else
                echo "Toolbox '$workspace_name' does not exist."
            fi

            echo "Removing image..."

            if podman image exists "$image_name"; then
                podman rmi "$image_name"
            else
                echo "Image '$image_name' does not exist."
            fi

            echo "Cleanup completed."
            break
            ;;

        n|N|"")
            echo "Cleanup cancelled."
            break
            ;;

        *)
            echo "Response not valid. Enter y or n."
            ;;

esac
done
