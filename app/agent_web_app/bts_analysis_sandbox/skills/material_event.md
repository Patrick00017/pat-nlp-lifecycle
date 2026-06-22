# 换材事件

## 事件格式

```python 
{
'id': uuid.uuid1(),
'part': part,
'msg': f"({prev_material_batch['material']},{prev_material_batch['width']},{prev_material_batch['flute_type']}) -> ({current_material_batch['material']},{current_material_batch['width']},{current_material_batch['flute_type']})",
'time': str(row['Date']),
'reason': 'reset'
}
```

id用于后续进行追踪，reason的值为normal或者reset，用于后续判断。

用于记录接纸事件的数据结构

```python
self.splicer_state = {
            'ls0': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            },
            'ms1': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            },
            'ls1': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            },
            'ms2': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            },
            'ls2': {
                'material': '-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            },
            'df': {
                'material': '-.-.-.-.-',
                'width': 0,
                'flute_type': '-',
                'next_batch': {
                    'material': '-.-.-.-.-',
                    'width': 0,
                    'flute_type': '-',
                },
                'change_time': '2026-03-02 15:39:32.530'
            }
        }
```