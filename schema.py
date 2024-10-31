TRAIN_SCHEMA = {
    'images': {
        'type': list,
        'required': True
    },
    'class_name': {
        'type': str,
        'required': False,
        'default': ""
    },
    'identifier': {
        'type': str,
        'required': False,
        'default': "sks"
    },
    "batch_size": {
        'type': int,
        'required': False,
        'default': 16
    },
    'steps': {
        'type': int,
        'required': False,
        'default': 250
    }
}
