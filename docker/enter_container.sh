#!/bin/bash
su_=$1

if [ "$su_" = "true" ]; then
    docker exec -it -u 0 torch_container_gui /bin/bash
else
    docker exec -it torch_container_gui /bin/bash
fi