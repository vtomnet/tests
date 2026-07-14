#!/bin/sh
set -eu

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

magick -size 1024x768 xc:white -fill red -draw "circle 751,275 756,275" "$dir/image.png"
