from tkinter import *
import time
from funcs import check_input, calculate_price

if __name__ == '__main__':
	
	#initiallizing labels and entries for gui
    root = Tk()
    output_label = Label(root, font='ariel')
    city_entry = Entry(root)
    address_entry = Entry(root)
    num_beds_entry = Entry(root)
    size_entry = Entry(root)
    floor_entry = Entry(root)
    yearBuilt_entry = Entry(root)


    #defining usful functions
    def input_handler():
        global output_label
        output_label.forget
        house_data = {'city': str(city_entry.get()), 'address': str(address_entry.get()), 'beds': num_beds_entry.get(),
                      'area': size_entry.get(), 'floor': floor_entry.get(), 'yearBuilt': yearBuilt_entry.get()}

        if check_input(house_data):
            status, price_text = calculate_price(house_data)		#checks if enough featuers are typed and calculates price
            if status:
                output_label['text'] = "The price of the house is : " + price_text +" million Nis"
                output_label['fg'] = 'blue'
                output_label.grid(row=7,columnspan=7)
            else:
                output_label['text'] = 'Somthing went wrong, try agein'
                output_label['fg'] = 'red'
                output_label.grid(row=7,columnspan=7)
        else:
            output_label['text'] = 'Please add more information.\nThe labels with * are must'
            output_label['fg'] = 'red'
            output_label.grid(row=7,columnspan=7)


    #defining input labels
    Header = Label(root, text = "House Price Predictor", font='ariel')
    city_label = Label(root, text = "* City:")
    address_label = Label(root, text = "Address:")
    num_beds_label = Label(root, text = "Number of bedrooms:")
    size_label = Label(root, text = "Size:")
    floor_label = Label(root, text = "Floor:")
    yearBuilt_label = Label(root, text = "Built Year:")
    Submit_Button = Button(root, text = "Submit", command=input_handler)
    Blank_row1 = Label(root, text = " ")
    Blank_row2 = Label(root, text=" ")

    #arranging widgets
    Header.grid(row=0, columnspan=7)
    Blank_row1.grid(row=1)
    labels=3
    entries=4
    city_label.grid(row=labels, column=0)
    city_entry.grid(row=entries, column=0)

    address_label.grid(row=labels, column=1)
    address_entry.grid(row=entries, column=1)

    num_beds_label.grid(row=labels, column=2)
    num_beds_entry.grid(row=entries, column=2)

    size_label.grid(row=labels, column=3)
    size_entry.grid(row=entries, column=3)

    floor_label.grid(row=labels, column=4)
    floor_entry.grid(row=entries, column=4)

    yearBuilt_label.grid(row=labels, column=5)
    yearBuilt_entry.grid(row=entries, column=5)
    Blank_row2.grid(row=5)

    Submit_Button.grid(row=entries,column=6)


    root.mainloop()