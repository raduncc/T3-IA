class Document:

    def __init__(self, content, summary, dtype):
        self.__content = content
        self.__summary = summary
        self.__dtype = dtype

    def show_info(self):
        print("Content is: " + self.get_content())
        print("Summary is: " + self.get_summary())
        print("Type is: " + self.get_dtype)


    def add_content(self, content):
        self.__content = content

    def add_summary(self, summary):
        self.__summary = summary

    def add_dtype(self, dtype):
        self.__dtype = dtype

    def get_content(self):
        return self.__content

    def get_summary(self):
        return self.__summary

    def get_dtype(self):
        return self.__dtype
