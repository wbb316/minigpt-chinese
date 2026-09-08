import sys
sys.path.insert(0, '/root')
from model.layers import FF_TYPES, swiglu_hidden
from model import ffn_adapter
print('FF_TYPES:', FF_TYPES)
print('swiglu_hidden(576):', swiglu_hidden(576))
print('ffn_adapter import OK')
