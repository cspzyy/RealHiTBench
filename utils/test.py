if __name__ == '__main__':
    from chart_metric_util import *
    import pandas as pd
    import matplotlib.pyplot as plt
    data = {'Bituminous (100%)': 2.411842, 'Black Pellets (20%)': 2.197341, 'Black Pellets (40%)': 1.65341, 'Charred Biomass (20%)': 2.009878, 'Charred Biomass (40%)': 1.496352, 'Dark Roast Biomass (20%)': 2.065801, 'Dark Roast Biomass (40%)': 1.578632, 'Light Roast Biomass (20%)': 1.920086, 'Light Roast Biomass (40%)': 1.608222}
    labels = sorted(data.keys())
    values = [data[label] for label in labels]
    print(labels)
    print(values)
    plt.bar(labels, values)
    plt.xlabel('Material and Blend')
    plt.ylabel('Average SO2 Inlet Value (lb/MMBtu)')
    plt.title('Comparison of Average SO2 Inlet Values for Material and Blend Combinations')
    plt.xticks(rotation=45)
    plt.show()
    
    chart_type="BarChart"
    
    if chart_type == 'LineChart': 
        y_predictions = get_line_y_predictions(plt)
    if chart_type == 'BarChart':
        y_predictions = get_bar_y_predictions(plt)
    if chart_type == 'ScatterChart':
        y_predictions = get_scatter_y_predictions(plt)
    if chart_type == 'PieChart':
        y_predictions = get_pie_y_predictions(plt)
    
    print(y_predictions)