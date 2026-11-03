project('framegen-compute-poc', 'cpp', default_options: ['cpp_std=c++20', 'warning_level=3'])

vulkan = dependency('vulkan', version: '>=1.2')
glslang = find_program('glslangValidator')
shader = custom_target('generation-shader', input: 'generate.comp', output: 'generate.spv',
  command: [glslang, '-V', '--target-env', 'vulkan1.2', '@INPUT@', '-o', '@OUTPUT@'])
inc = include_directories('../../src')
engine = static_library('generation-engine', 'vulkan_engine.cpp',
  include_directories: inc, dependencies: vulkan)
probe = executable('framegen-compute-poc', 'main.cpp',
  include_directories: inc, link_with: engine, dependencies: vulkan,
  link_depends: shader,
  cpp_args: ['-DFRAMEGEN_COMPUTE_SHADER="' + (meson.current_build_dir() / 'generate.spv') + '"'])
contracts = executable('compute-contracts', 'test_contracts.cpp', include_directories: inc)
test('compute-contracts', contracts)
# GPU checks are explicitly invoked, never part of an automatic Meson test.

python = find_program('python3')
backend_contracts = executable('backend-contracts', 'test_backend_contract.cpp', include_directories: inc)
test('backend-contracts', backend_contracts)
test('backend-python-contracts', python,
  args: [files('../../scripts/test_framegen_backend.py'), '--contract-binary', backend_contracts])
training_abi = executable('training-abi-contracts', 'test_training_abi.cpp', include_directories: inc)
test('training-abi-contracts', training_abi)
test('training-consumer-contracts', python, args: files('test_training.py'), timeout: 30)
test('nr-probe-contracts', python, args: files('nr/test_nr.py'), timeout: 30)
training_sources = custom_target('training-reference-sources',
  input: ['../../src/shaders/cs_framegen_motion_net_train.comp',
          '../../src/shaders/cs_framegen_motion_net_opt.comp',
          '../../src/shaders/framegen_evidence.h'],
  output: ['training-reference-grad.comp', 'training-reference-opt.comp'],
  command: [python, files('make_training_reference.py'), '@INPUT@', '@OUTPUT@'])
training_grad = custom_target('training-reference-grad', input: training_sources[0],
  output: 'training-reference-grad.spv',
  command: [glslang, '-V', '--target-env', 'vulkan1.2', '-S', 'comp', '@INPUT@', '-o', '@OUTPUT@'])
training_opt = custom_target('training-reference-opt', input: training_sources[1],
  output: 'training-reference-opt.spv',
  command: [glslang, '-V', '--target-env', 'vulkan1.2', '-S', 'comp', '@INPUT@', '-o', '@OUTPUT@'])
training_reference = executable('framegen-training-reference', 'training_reference.cpp',
  'training_fixture.cpp', include_directories: inc, link_with: engine, dependencies: vulkan,
  link_depends: [training_grad, training_opt],
  cpp_args: ['-DFRAMEGEN_TRAIN_GRAD_SHADER="' + (meson.current_build_dir() / 'training-reference-grad.spv') + '"',
             '-DFRAMEGEN_TRAIN_OPT_SHADER="' + (meson.current_build_dir() / 'training-reference-opt.spv') + '"'])

report_contracts = executable('training-report-contracts', 'test_training_report.cpp',
  include_directories: inc)
test('training-report-contracts', report_contracts)
resident_health = custom_target('resident-health', input: 'resident_health.comp',
  output: 'resident-health.spv',
  depend_files: files('../../src/framegen/training_report.h'),
  command: [glslang, '-V', '--target-env', 'vulkan1.2', '-I' + (meson.current_source_dir() / '../../src'),
            '@INPUT@', '-o', '@OUTPUT@'])
resident_publish = custom_target('resident-publish', input: 'resident_publish.comp',
  output: 'resident-publish.spv',
  depend_files: files('../../src/framegen/training_report.h'),
  command: [glslang, '-V', '--target-env', 'vulkan1.2', '-I' + (meson.current_source_dir() / '../../src'),
            '@INPUT@', '-o', '@OUTPUT@'])
resident_args = ['-DFRAMEGEN_TRAIN_GRAD_SHADER="' + (meson.current_build_dir() / 'training-reference-grad.spv') + '"',
                 '-DFRAMEGEN_TRAIN_OPT_SHADER="' + (meson.current_build_dir() / 'training-reference-opt.spv') + '"',
                 '-DFRAMEGEN_RESIDENT_HEALTH_SHADER="' + (meson.current_build_dir() / 'resident-health.spv') + '"',
                 '-DFRAMEGEN_RESIDENT_PUBLISH_SHADER="' + (meson.current_build_dir() / 'resident-publish.spv') + '"']
resident = static_library('resident-training', 'resident_training.cpp',
  include_directories: inc, dependencies: vulkan, cpp_args: resident_args,
  link_depends: [training_grad, training_opt, resident_health, resident_publish])
resident_health_test = executable('framegen-resident-health', 'test_resident_health.cpp',
  include_directories: inc, link_with: engine, dependencies: vulkan, cpp_args: resident_args,
  link_depends: [resident_health, resident_publish])
resident_replay = executable('framegen-resident-training', 'resident_main.cpp', 'training_fixture.cpp',
  include_directories: inc, link_with: [resident, engine], dependencies: vulkan, cpp_args: resident_args,
  link_depends: [training_grad, training_opt, resident_health, resident_publish])
resident_test_opt = custom_target('resident-test-opt', input: 'resident_test_optimizer.comp',
  output: 'resident-test-opt.spv', depend_files: files('../../src/framegen/training_report.h'),
  command: [glslang, '-V', '--target-env', 'vulkan1.2', '-I' + (meson.current_source_dir() / '../../src'),
            '@INPUT@', '-o', '@OUTPUT@'])
resident_test_health_source = custom_target('resident-test-health-source', input: 'resident_health.comp',
  output: 'resident-test-health.comp',
  command: [python, files('make_resident_test_health.py'), '@INPUT@', '@OUTPUT@'])
resident_test_health = custom_target('resident-test-health', input: resident_test_health_source,
  output: 'resident-test-health.spv', depend_files: files('../../src/framegen/training_report.h'),
  command: [glslang, '-V', '--target-env', 'vulkan1.2', '-I' + (meson.current_source_dir() / '../../src'),
            '@INPUT@', '-o', '@OUTPUT@'])
resident_owner_test = executable('framegen-resident-owner', 'test_resident_owner.cpp', 'resident_training.cpp',
  include_directories: inc, link_with: engine, dependencies: vulkan,
  cpp_args: [resident_args[0], resident_args[3],
             '-DFRAMEGEN_TRAIN_OPT_SHADER="' + (meson.current_build_dir() / 'resident-test-opt.spv') + '"',
             '-DFRAMEGEN_RESIDENT_HEALTH_SHADER="' + (meson.current_build_dir() / 'resident-test-health.spv') + '"'],
  link_depends: [training_grad, resident_test_opt, resident_test_health, resident_publish])

if not get_option('hip_training').disabled()
  hipcc = find_program('hipcc', dirs: ['/opt/rocm/bin'], required: get_option('hip_training'))
  if hipcc.found()
    arch = get_option('hip_arch')
    if arch == ''
      error('Set -Dhip_arch explicitly; building must not enumerate or choose a GPU')
    endif
    hip_plugin = custom_target('hip-training-plugin',
      input: ['hip_training.cpp', 'hip_training_kernels.hip'],
      output: 'libframegen-training-hip.so', build_by_default: true,
      depend_files: files('training_plugin.h', 'training_kernels.hpp',
                         '../../src/framegen/compute_job.hpp', '../../src/framegen/net_layout.hpp',
                         '../../src/framegen/backend_contract.h', '../../src/framegen/backend_contract.def'),
      command: [hipcc, '-std=c++20', '-O2', '-fPIC', '-shared', '-ffp-contract=off',
                '--offload-arch=' + arch, '-I' + (meson.current_source_dir() / '../../src'),
                '@INPUT@', '-o', '@OUTPUT@'])
  endif
endif
