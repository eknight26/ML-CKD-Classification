import sys

def error_msg_detail(error, error_detail: sys):
    # Extracts the file name and line number where the error occurred.
    _, _, exc_tb = error_detail.exc_info()
    
    # Safety check in case traceback is None
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown Script"
        line_number = "Unknown"

    error_message = f"Error occurred in script: [{file_name}] line number: [{line_number}] error message: [{str(error)}]"
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # We store the detailed message formatted by our helper function
        self.error_message = error_msg_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    