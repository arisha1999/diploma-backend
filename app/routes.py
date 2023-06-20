from app.controllers.audio_controller import AudioController

ROUTES = [
    {
        'prefix': '/api',
        'group': [
            {'prefix': '/audio', 'controller':  AudioController},
        ]
    }
]