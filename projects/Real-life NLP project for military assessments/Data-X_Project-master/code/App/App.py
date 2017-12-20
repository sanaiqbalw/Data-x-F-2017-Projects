from tkinter import *
from bullet_rules import *
from converter import RunConvert
from tkinter import messagebox

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

        #make checkbox that signals synonym flags (#)
        self.syn_flag = IntVar()
        self.check = Checkbutton(self.frame, text="Synonym Flag (#)", var=self.syn_flag)
        self.check.grid(row=2, column=3)
        #make it default to on
        self.check.select()

        #make a text box to input the converted text
        self.converted_text = Entry(self.frame, width=100, font="Times 13")
        self.converted_text.grid(row=3,columnspan=3, column=1, sticky=W)
        # self.converted_button =  Button(self.frame, width = 7, text="Copy", command=self.store_out,
                                # font='Courier')
        # self.converted_button.grid(row=3, column=3, sticky=E)

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
        convert_ouput = RunConvert(string, self.syn_flag)   #full version
        #bullets = [BulletRules(string)] #demo version

        if convert_ouput == -1:
            #original bullet too short
            messagebox.showinfo("Too Short", "The bullet is too short to fill the line.")
        elif convert_ouput == -2:
            #original bullet too long
            messagebox.showinfo("Too Long", "The bullet is too long to fit the line.")
        else:
            #display the converted bullets
            self.bullet_out = convert_ouput
            self.display_output()

    def display_output(self):
        #first check to see if the scroll button is already there
        try:
            self.more_bull
        except AttributeError:
            button_exists = False
        else:
            button_exists = True

        #create/delete the scroll button if necessary
        if (len(self.bullet_out) > 5) and (button_exists==False):
            #if there are more than 5 bullet options, create the scroll button
            self.more_bull = Button(self.frame, width = 10, text="Show More", 
                                    command=self.display_output, font='Courier')
            self.more_bull.grid(row=10, column=3)
        elif (len(self.bullet_out) <= 5) and button_exists:
            #if there are <= 5 bullet options left, get rid of the scroll button
            self.more_bull.grid_forget()

        #output converted text
        self.converted_text.delete(0, END)
        self.converted_text.insert(0, self.bullet_out[0])
        #show up to 4 more bullet options
        self.bullet = []
        i=0
        for i in range(1, min(len(self.bullet_out), 5)):
            self.bullet.append(Entry(self.frame, width=100, font="Times 13"))
            self.bullet[i-1].grid(row=3+i, columnspan=3, column=1, sticky=W)
            self.bullet[i-1].insert(0, self.bullet_out[i])

        
        #remove bullets already output to user
        self.bullet_out = self.bullet_out[i+1:]

    def exit(self):
        self.frame.quit()
        root.destroy()
    
    def store_out(self):
        pass


root = Tk()
root.title("Bullet Converter")
app = App(root)

root.mainloop()