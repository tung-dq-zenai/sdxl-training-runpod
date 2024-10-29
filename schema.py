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
    }
}
