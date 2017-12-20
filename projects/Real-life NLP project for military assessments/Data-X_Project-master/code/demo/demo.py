from tkinter import *
from bullet_rules import *

class App:
    '''
    This class defines the bullet converter gui
    '''
    def __init__(self, master):
        #initialization steps
        self.frame = Frame(master, width = 120)
        self.frame.pack()

        #make a quit button
        self.button = Button(self.frame, width = 7, text="Quit", fg="red",
                                command=self.exit, font='Courier')
        self.button.grid(row=10, column=2)

        #make a convert button
        self.con_but = Button(self.frame, width = 7, text="Convert", command=self.convert,
                                font='Courier')
        self.con_but.grid(row=10, column=1)
        
        #make a user entryboxes (1 for each part of bullet)
        self.action = Entry(self.frame, width=50, font="Times 13")
        self.action.grid(row=1, column=1)
        self.action.bind("<Return>", self.convert)

        self.impact = Entry(self.frame, width=50, font="Times 13")
        self.impact.grid(row=1, column=3)
        self.impact.bind("<Return>", self.convert)
        
        self.result = Entry(self.frame, width=50, font="Times 13")
        self.result.grid(row=2, column=1)
        self.result.bind("<Return>", self.convert)

        #make a text box to input the converted text
        self.converted_text = Entry(self.frame, width=100, font="Times 13")
        self.converted_text.grid(row=3,columnspan=3, column=1, sticky=W)

        #label text boxes
        self.action_label = Label(self.frame, text="Action", font="Courier")
        self.action_label.grid(row=1, column=0)
        self.impact_label = Label(self.frame, text="Impact", font="Courier")
        self.impact_label.grid(row=1, column=2, sticky=E)
        self.result_label = Label(self.frame, text="Result", font="Courier")
        self.result_label.grid(row=2, column=0)
        self.output_label = Label(self.frame, text="Bullet", font="Courier")
        self.output_label.grid(row=3, column=0)

    def convert(self, event=None):
        '''
        This function converts the user's input using the Substitutions function and prints the converted
        text to the second text box in the interface.
        '''
        #get user input
        action_string = self.action.get()
        impact_string = self.impact.get()
        result_string = self.result.get()
        #combine to create one string
        string =  action_string + '; '*bool(impact_string) + impact_string + '--'*bool(result_string) + result_string 
        
        #perform conversion
        #bullets = RunConvert(string)   #full version
        bullets = [BulletRules(string)] #demo version

        #output converted text
        self.converted_text.delete(0, END)
        self.converted_text.insert(0, bullets[0])
        #show all bullet options
        self.bullet = []
        for i in range(len(bullets)):
            self.bullet.append(Entry(self.frame, width=100, font="Times 13"))
            self.bullet[i].grid(row=4+i, columnspan=3, column=1, sticky=W)
            self.bullet[i].insert(0, bullets[i])

    def exit(self):
        self.frame.quit()
        root.destroy()


root = Tk()
root.title("Bullet Converter")
app = App(root)

root.mainloop()