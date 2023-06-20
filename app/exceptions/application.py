from app.exceptions import AppException

class ApplicationNotFoundException(AppException):
    def __init__(self):
        self.message = 'Landing not found'

class ApplicationFolderNotFoundException(AppException):
    def __init__(self):
        self.message = 'Landing folder not found'

class LandingNameAlreadyExists(AppException):
    def __init__(self):
        self.message = 'Landing name already exists'