catch2_dep = dependency('catch2-with-main', required: true)

srcdir = '../src'
unittest_src = []
foreach src : gamescope_core_src + ['Script/Script.cpp', 'convar_script.cpp']
	unittest_src += srcdir / src
endforeach

gamescope_tests = executable(
	'gamescope_tests',
	unittest_src + ['../src/color_helpers.cpp', 'test_color_helpers.cpp', 'test_convar.cpp', 'test_focus_placement.cpp', 'test_mangoapp_config.cpp', 'test_mangoapp_control.cpp', 'test_mode_list.cpp', 'test_present_timing_schedule.cpp', 'test_process.cpp', 'test_swapchain_wire.cpp', 'test_upscale.cpp', 'test_utils_commit_queue.cpp', 'test_utils_parsers.cpp', 'test_utils_string.cpp', 'test_utils_vblank_schedule.cpp', 'test_vrclient_detect.cpp'] + swapchain_client_src + limiter_client_src + [swapchain_server_header],
	include_directories: [srcdir, sol2_include],
	dependencies: [catch2_dep, luajit_dep, cap_dep, drm_dep, glm_dep, vulkan_dep, wlroots_dep, dep_wayland, wayland_server],
	cpp_args: gamescope_cpp_args,
)

reaper_test_helper = executable('reaper_test_helper', 'reaper_test_helper.cpp')

test('convar', gamescope_tests, args: ['[convar]'])
test('color_helpers', gamescope_tests, args: ['[color_helpers]'])
test('focus_placement', gamescope_tests, args: ['[focus_placement]'])
test('mangoapp_config', gamescope_tests, args: ['[mangoapp_config]'])
test('mangoapp_control', gamescope_tests, args: ['[mangoapp_control]'])
test('mode_list', gamescope_tests, args: ['[mode_list]'])
test('present_timing_schedule', gamescope_tests, args: ['[present_timing_schedule]'])
test('swapchain_wire', gamescope_tests, args: ['[swapchain_wire]'])
test('upscale', gamescope_tests, args: ['[upscale]'])
test('utils/commit_queue', gamescope_tests, args: ['[commit_queue]'])
test('utils/parsers', gamescope_tests, args: ['[parsers]'])
test('utils/process', gamescope_tests, args: ['[process]'], depends: [reaper_test_helper])
test('utils/string', gamescope_tests, args: ['[string]'])
test('utils/vblank_schedule', gamescope_tests, args: ['[vblank_schedule]'])
test('vrclient_detect', gamescope_tests, args: ['[vrclient_detect]'])

if get_option('enable_gamescope_wsi_layer')
	present_timing_layer_tests = executable(
		'present_timing_layer_tests',
		['test_present_timing.cpp', 'test_present_timing_layer.cpp', protocols_client_src, swapchain_server_header],
		dependencies: [catch2_dep, vkroots_dep, vulkan_dep, dep_xcb, dep_x11, dep_x11_xcb, glm_dep, wayland_client, wayland_server],
	)
	test('present_timing', present_timing_layer_tests, args: ['[present_timing]'])
	test('present_timing_layer', present_timing_layer_tests, args: ['[present_timing_layer]'])
endif

test_framegen = executable(
	'test_framegen',
	['test_framegen.cpp', srcdir / 'framegen/atomic_file.cpp'],
	include_directories: [srcdir],
	cpp_args: gamescope_cpp_args,
)
test('framegen', test_framegen)

test_framegen_deadline = executable(
	'test_framegen_deadline',
	['framegen_deadline.cpp'],
	include_directories: [srcdir],
	dependencies: [catch2_dep],
	# The HUD tests compare hud.hpp with the shader source itself, and pin the
	# HUD wiring in rendervulkan.cpp, which no test compiles.
	cpp_args: gamescope_cpp_args + [
		'-DFRAMEGEN_HUD_SHADER_PATH="@0@"'.format(
			meson.project_source_root() / 'src' / 'shaders' / 'cs_framegen_hud.comp'),
		'-DFRAMEGEN_RENDERVULKAN_SOURCE_PATH="@0@"'.format(
			meson.project_source_root() / 'src' / 'rendervulkan.cpp'),
	],
)
test('framegen_deadline', test_framegen_deadline)

test('wlserver_shutdown', find_program('python3'),
	args: files('test_wlserver_shutdown.py'), timeout: 90)

test(
	'gpu_diagnostics',
	find_program('python3'),
	args: files('../tools/vapor-menu/test_gpu_diagnostics.py'),
	timeout: 360,
)

test_swapchain_policy = executable(
	'test_swapchain_policy',
	['test_swapchain_policy.cpp'],
	include_directories: [srcdir],
	dependencies: [catch2_dep, vulkan_dep.partial_dependency(compile_args: true, includes: true)],
	cpp_args: gamescope_cpp_args,
)
test('swapchain_policy', test_swapchain_policy)

test_vblank_grid = executable(
	'test_vblank_grid',
	['test_vblank_grid.cpp'],
	include_directories: [srcdir],
	dependencies: [catch2_dep],
	cpp_args: gamescope_cpp_args,
)
test('vblank_grid', test_vblank_grid)
