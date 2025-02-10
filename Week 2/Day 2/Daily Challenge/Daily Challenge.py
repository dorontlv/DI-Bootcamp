'''

Daily Challenge : Pagination

What You will learn :
OOP

'''


class Pagination:
    def __init__(self, items=None, pageSize=10):
        self.items = items

        pageSize = int(pageSize)
        self.pageSize = pageSize
        
        self.location = 0  # I'm not using pages (currentPage and totalPages) but the exact location in the text
        self.size = len(items)


    def getVisibleItems(self):
        print([item for item in self.items[self.location:self.location+self.pageSize]])  # string comprehension
        print("")

    def prevPage(self):
        if self.location - self.pageSize >= 0:
            self.location -= self.pageSize

    def nextPage(self):
        if self.location + self.pageSize < self.size:
            self.location += self.pageSize
        return self  # so that the method is chainable.

    def firstPage(self):
        self.location = 0  # just go to 0

    def lastPage(self):
        if self.size % self.pageSize == 0:  # check if it exactly devides by the page size
            self.goToPage(int(self.size / self.pageSize))
        else:
            self.goToPage(int(self.size / self.pageSize)+1)

    def goToPage(self, pageNum):
        if pageNum<1:
            return
        pageNum = int(pageNum)  # it might be a float number
        if (pageNum-1) * self.pageSize < self.size:
            self.location = (pageNum-1) * self.pageSize  # calculate the page number
        else:
            self.lastPage()  #  it's beyond the text size, so just go to the last page

            

alphabetList = list("abcdefghijklmnopqrstuvwxyz")

p = Pagination(alphabetList, 4)

p.getVisibleItems() 
# ["a", "b", "c", "d"]

p.nextPage()
p.getVisibleItems()
# ["e", "f", "g", "h"]

p.prevPage()
p.getVisibleItems() 

p.lastPage()
p.getVisibleItems()
# ["y", "z"]

p.nextPage()
p.getVisibleItems() 

p.prevPage()
p.getVisibleItems()

p.firstPage()
p.getVisibleItems()

p.prevPage()
p.getVisibleItems()

p.goToPage(1)
p.getVisibleItems()

p.goToPage(2)
p.getVisibleItems()

p.goToPage(10)
p.getVisibleItems()

p.firstPage()
p.getVisibleItems()

p.nextPage().nextPage().nextPage()
p.getVisibleItems()

