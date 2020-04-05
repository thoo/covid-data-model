# TFFN (Talus Final For Now)
## Rewrite of the model run approach and addition of asymptomatic class

Super-excited that [Eric](https://github.com/erccarls) is taking over and will be able to
work on this full time and give it the attention it deserves. That said, there were a features
and changes in the model that we wanted to get in for our own purposes and mostly
just to get everything out of our heads before moving on to other focuses.

Understanding this will probably never feed the covidactnow.org site, these changes
make it easier to tackle one-off questions using the model (e.g. impact of current state interventions)
and so were useful for the Talus/GU team. Thought they might also help the broader team.

A lot of stuff in here focused on 3 main aims:
* General campsite rule cleanup and documentation
* Introduction of TalusSEIR model with asymptomatic class. This is very similar
to Eric's Epi model with the only difference (I think) being that people do not
die out of the mild or hospitalized stock in this model
* (Nearly finished) Much more OO approach to model runs that makes it easier to follow
what's going on with interventions - will finalize this in the next day or so
