generate:
	@echo "Generating data for Gephi"
	deno run --allow-read --allow-write main.ts
	@echo "data generated"
	@echo "moving the data to ~/Music/gephi-chart"
	mv edges.csv ~/Music/gephi-chart
	mv nodes.csv ~/Music/gephi-chart

open:
	@echo "opeing the gephi app"
	~/Downloads/gephi-0.10.1-linux-x64/gephi-0.10.1/bin/gephi