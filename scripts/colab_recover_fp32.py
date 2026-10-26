# Paste into a new Colab code cell after rerunning setup/configuration cells.
# Uses the existing model cache; preserves the old activation folder.
from pathlib import Path
import yaml
from intrinsic_dim.activations import extract

config.device = 'cuda'
config.dtype = 'float32'
config.batch_size = 1
old_output = Path(config.output_dir)
OUTPUT = old_output if old_output.name.endswith('_fp32') else old_output.with_name(old_output.name + '_fp32')
config.output_dir = str(OUTPUT)
config.validate()
RUN_CONFIG.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))

print('Retrying entire extraction in CUDA float32:', OUTPUT)
metadata = extract(config)  # Original exception is displayed directly.
assert metadata['complete']
print('Extraction complete:', metadata['completed_rows'], 'prompts')
print('Actual device/dtype:', metadata['model']['device'], metadata['model']['dtype'])
# Continue with notebook steps 7 and 8 using this RUN_CONFIG and OUTPUT.
