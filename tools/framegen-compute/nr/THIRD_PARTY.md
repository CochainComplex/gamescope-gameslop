# Third-party source notices

These notices cover the upstream source portions used by this adapter. They do
not replace the licences of separately supplied models, runtime binaries or
their dependencies. See [upstream projects and attribution](README.md#upstream-projects-and-attribution)
for the experiment's dependency references and roles.

## RedDukeDev DLSS 5 NR AMD Custom

`prepare.py` contains patch anchors from, and prepares modified private copies of,
the runtime and harness at [revision b5744cf03208f1442d63eeacc48eaf77a6bb4765](https://github.com/RedDukeDev/dlss5-nr-amd-custom/commit/b5744cf03208f1442d63eeacc48eaf77a6bb4765).
Device selection and correctness diagnostics are local adaptations. The source
notice below is retained verbatim from that revision's `LICENSE`; its paths
refer to the upstream project.

```text
MIT License

Copyright (c) 2026 RedDukeDev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This licence covers kernels/, runtime/ and docs/. The optiscaler/ submodule is
a fork of OptiScaler and is licensed under the GNU GPL version 3, as its
upstream is. third_party/ZLUDA keeps its own licence (MIT or Apache-2.0).
```

## DXVK

The COM method declarations and interface identifiers in `presenter_identity.hpp`
are reduced and renamed adaptations of DXVK's
[`src/dxgi/dxgi_interfaces.h`](https://github.com/doitsujin/dxvk/blob/master/src/dxgi/dxgi_interfaces.h).
The Vulkan UUID/PCI selection and diagnostics around them are local code. The
following notice is retained verbatim from `dxvk/LICENSE` in the pinned RedDukeDev
source, also matching [DXVK's licence](https://github.com/doitsujin/dxvk/blob/master/LICENSE).

```text
                  Copyright (c) 2017 Philip Rebohle
                  Copyright (c) 2019 Joshua Ashton
                  Copyright (c) 2019 Robin Kertels
                  Copyright (c) 2023 Jeffrey Ellison

                          zlib/libpng license

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

– The origin of this software must not be misrepresented; you must not
  claim that you wrote the original software. If you use this software
  in a product, an acknowledgment in the product documentation would be
  appreciated but is not required.

– Altered source versions must be plainly marked as such, and must not
  be misrepresented as being the original software.

– This notice may not be removed or altered from any source distribution.
```
