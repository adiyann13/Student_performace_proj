import sys
import os

class Custom_Exception(Exception):
    def __init__(self,eror_mssg,error_detail:sys):
        self.eror_mssg = eror_mssg
        _,_,exec_tb = error_detail.exc_info()
        self.line_no = exec_tb.tb_lineno
        self.file_name = exec_tb.tb_frame.f_code.co_filename
    
    def __str__(self):
        return f' error/exception mssg : {str(self.eror_mssg)}occured in line no {self.line_no} of file {self.file_name}'
