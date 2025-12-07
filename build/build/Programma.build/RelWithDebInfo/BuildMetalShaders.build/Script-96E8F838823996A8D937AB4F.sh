#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/rubenvlieger/Documents/RVGRT/build
  /usr/bin/xcrun metal -c -frecord-sources -gline-tables-only -I /Users/rubenvlieger/Documents/RVGRT/include /Users/rubenvlieger/Documents/RVGRT/src/renderer/kernels/exposure_kernel.metal -o /Users/rubenvlieger/Documents/RVGRT/build/exposure_kernel.air
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/rubenvlieger/Documents/RVGRT/build
  /usr/bin/xcrun metal -c -frecord-sources -gline-tables-only -I /Users/rubenvlieger/Documents/RVGRT/include /Users/rubenvlieger/Documents/RVGRT/src/renderer/kernels/exposure_kernel.metal -o /Users/rubenvlieger/Documents/RVGRT/build/exposure_kernel.air
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/rubenvlieger/Documents/RVGRT/build
  /usr/bin/xcrun metal -c -frecord-sources -gline-tables-only -I /Users/rubenvlieger/Documents/RVGRT/include /Users/rubenvlieger/Documents/RVGRT/src/renderer/kernels/exposure_kernel.metal -o /Users/rubenvlieger/Documents/RVGRT/build/exposure_kernel.air
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/rubenvlieger/Documents/RVGRT/build
  /usr/bin/xcrun metal -c -frecord-sources -gline-tables-only -I /Users/rubenvlieger/Documents/RVGRT/include /Users/rubenvlieger/Documents/RVGRT/src/renderer/kernels/exposure_kernel.metal -o /Users/rubenvlieger/Documents/RVGRT/build/exposure_kernel.air
fi

