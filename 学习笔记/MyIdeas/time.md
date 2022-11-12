# 【好文摘录】Time management for the single-threaded mind

[原文在此](https://shengchen.design/Time-management-for-the-single-threaded-mind/)

Over the years, I gradually accepted that at any given time, I'd always have multiple projects at hand eager for my attention. I have course projects, lab(s) research, hobby projects, random bursts of passionate fixations… I also gradually learned that a project eventually dies if left alone for long enough, when it loses all that glamor it once had when you first started it.

However, juggling multiple projects at once is never easy. They say that our minds are inherently single-threaded, no matter how good we seem at “multitasking.” It takes effort to work on a project, but it also takes effort to decide *when* and *how* to work on it. Well, it even takes up space and effort to remember every project you're working on!

Being the type of person that gets completely absorbed when working on something, it can be hard to keep taking a step back, remember everything I did and should do, and sort out all the priorities. Doing so breaks my “flow.” Sometimes, it gives me a headache just to decide what to do on a given day.

I've come to learn that sometimes, the real enemy is not “too many projects.” The work might be manageable, yet still suffocating. The real enemy is the anxiety, constant mental load, frustration, and guilt that comes from having to constantly switch tasks, needing to focus, yet still trying to keep an eye on what should happen next (and failing to do any of these things). The real enemy is fatigue.

Therefore, I thought: why not design an automated system that takes care of managing my projects? I want an “algorithm” that tells me when to pay attention to *what* project so that I never stray away for too long, while making sure that more important and urgent projects get more attention. At the same time, the system should also leave enough room for me to plan out the details of each project, and make adjustments whenever I need.

So I present to you this system after some trial and error — what I call *Cicada's Method* of time management. I find this method particularly suitable for managing multiple research projects alongside personal ones, where unexpected things can happen in any given project, and the plan changes constantly.

## The core idea

The philosophy of this method is delegating the work of *allocating your attention*, instead of helping you micro-manage deadlines and to-do list items. The system I set up tells me “when” I should pay attention to a certain project, but not “for how long” and “finish which tasks.” Roughly speaking, it's kind of like having a personal assistant that handles your schedule.

Now for some core ideas of this method:

### Attention, not deadlines

As mentioned above, this method shifts from managing deadlines to managing your attention. Honestly, you only have so much time, and you only have so much attention span. If you take on too many things at once, it's simply impossible to finish everything before their deadlines.

Therefore, stop worrying about them! Instead, make a list of your projects. Estimate how many work days it'll take for you to finish each project. It can be a very rough guess at the start, it doesn't matter. Your plans will change anyway, once you realize you cannot finish everything in time.

If you have no idea how many days the project will take you, you can still take a guess, even take a random number. It's a less educated guess, but a guess no less. It will become more accurate once you start working on the project.

Write the number of days down next to your project. Then, count how many days you have until the deadline. Divide it by the number of days and round it down, and you have an initial “attention period.” If your project doesn't have a deadline, congratulations! 😉 You can just randomly assign an attention period to the project however you like.

For example, if there's a project that you think takes 8 full days to finish, and needs to be done in 30 days, then its attention period is 30 ÷ 8 = 3.75 ≈ 3 days.

Overall, this attention period tells you that you'll have to work on the project once every that many days to have it finished by its deadline. But it also means that **you can completely forget about that project in between those days!**

### Cicada timing

Having an attention period sounds a bit more reassuring already, no? You get to completely forget about a project for some days. Way to free up that precious brain space!

But don't start using the schedule just yet… You might realize that you have a lot of projects with the same attention period. If you start now, you may find yourself drowning in projects on certain days where they overlap.

How do we solve this problem? Well, mother nature has already given us the answer. This problem has long been solved by the creature that gave this method its name — cicadas. Maybe you have already read an article about cicadas using prime numbers to avoid being eaten. But don't worry if you haven't. I'll give a very quick illustration of how the theory works[1].

>[1] You can also check out [this great video](https://www.youtube.com/watch?v=ivQaJwFRowc) from PBS Digital Studios for a more detailed, illustrated explanation!

As with many species, the population of cicadas and their predators change over time. In fact, their population oscillates with a certain number of years in between each peak.

Now, if cicadas and their predators both reach a peak in population in the same year, the cicadas would suffer drastic losses. Moreover, there is more than one type of predator! Let's say they each have a period of 2, 3, and 4 years. If some cicadas awake every 12 years, they would get easily eaten by all three types of predators!

```
Day	1	2	3	4	5	6	7	8	9	10	11	12
P1		X		X		X		X		X		X
P2			X			X			X			X
P3				X					X			X
Cicadas											X
```

Mathematically speaking, this is because 2, 3, and 4 are factors of 12. So what kinds of cicadas would avoid these predators? Those with a prime number period such as 13! They would only meet a peak of the 2-year predator every 26 years, the 3-year predator every 39 years, and the 4-year predator every 52 years! Therefore, most surviving cicadas have a period of a prime number.

What does this mean for our time management? —That we're better off choosing prime numbers as the attention period! This way, we avoid project collisions a lot better. For example, if I have 5 projects and I assign them attention periods 2, 3, 5, 7, and 11, then I shift the schedules a little, then I can guarantee that I do not need to pay attention to more than two projects on a given day!

Here's a chart showing the above example work in 12 days. You even get some free days!

```
Day	    1	2	3	4	5	6	7	8	9	10	11	12
P1		    X		X		X		X		X		X
P2		    	X			X			X			X
P3		    			X					X		
P4							X					
P5											X	
Total	0	1	1	1	1	2	1	1	1	2	1	2
```

So go through your list, and think about changing the attention periods a little bit, preferably to the nearest prime number.

### Constant adjustments

As promising as it sounds, cicada timing cannot guarantee a collision-free schedule. Maybe you're suddenly caught up in an emergency, therefore didn't have time to work on a project. Maybe you over- or under-estimated the days needed to finish the project. Maybe you just don't feel like working today… All of these require constant adjustment to both your plan and the attention periods.

#### Adjusting your plan

This might be the single most important thing to remember when adjusting your plans: plans are not set in stone. Even if they were, they might still be some ways around them. And if you truly cannot change the plans, well, these are the truly urgent projects… There's nothing to be done but to pay more attention to them!

But I digress. I find that reminding myself that constantly changing my plan is an option really takes away the stress. It helps me maintain a clear head and not get overwhelmed.

Therefore, keep changing the tasks, plans, milestones, and even personally-set deadlines. I always go through my to-do list for the project before I start working on it. I try to assess how far ahead (not often) or behind I am, and change my dates accordingly.

#### Adjusting the attention period

The best time to adjust the attention period is whenever it's time to work on a project. Just make it a habit that before you start working, first check if the project's attention period needs to be adjusted. You can re-do an estimation, or you can take a more educated guess after you've worked on it for some time.

Why only adjust the attention period on its working day? Here's why:

1. It's easier on the brain, since the entire goal is not paying undue attention.
2. It makes projects adjusted to have the same period unlikely to collide.

Let's look at my five projects again. Let's say that I drastically underestimated the time required by project five, and it needs to have a five-day attention period. However, I already have a five-day project (P3)! Will they collide?

```
Day	    11	12	13	14	15	16	17	18	19	20	21	22
P1	    	X		X		X		X		X		X
P2	    	X			X			X			X	
P3		    			X					X		
P4	    			X							X	
P5	    X					X					X	
Total	1	2	0	2	2	2	0	2	0	2	3	1
```

No, they will not. Because P5 started on day 11, there's no overlap with P3 until day 55, then once every 55 days. Unless you happen to change plans on those days (1 in 55 chance), you're safe. Besides, you can always choose a non-prime number, or shift the schedule around!

### No promises (to self)!

As demanding as this system sounds, the ultimate control is still yours. The schedule doesn't dictate what you do on a given day, just that you should pay at least some attention to the project. You didn't really make any promises that you'll finish anything on any day. Therefore, if you really can't work on a project on a given day, don't. Fatigue is the ultimate enemy here.

However, you still have to pay minimal attention: going through your original plan. Delete this day from it. Try to change the plan so that you can make up for it some other day. And what if you can't find the time? Then see if you can change the deadline. What if you can't change the deadline? Welp, then either ask for help [2], or… just do the project anyway, then make sure to take whatever measures you need to regain the energy you spent.

>[2] Note to self: remember that this is an option!!

## Implementing this system

But enough talk about abstract concepts! How can this system be implemented? You'll find that this system works really well together with a to-do list app, and is really easy to implement.

### Using to-do lists

Personally, I'm using [Things 3](https://culturedcode.com/things/). I have a friend who adopts this system that uses [TickTick](https://ticktick.com/). I've previously prototyped this system in [Notion](https://www.notion.so/) and even Numbers/Excel with formulas. Ultimately, I find that any to-do list app that supports repeating to-dos (for the attention period) will work well.

What I do is maintain a to-do list for every project I'm working on. I put all my detailed tasks inside the list as usual, but I don't really set dates for them. The one key thing is that I add a special to-do at the top of the list — one called “look at me! (project name/emoji)” I set it on repeat, with the period being the project's attention period.

Then, everything is done. Whenever there's a “look at me” to-do on my “today” page, I work on the project that it comes from. I begin by checking and adjusting the plan, then I check the attention period, then I start doing whatever in the list that needs to be done. After I finish a task, I tick it off the list as usual. It's a really smooth workflow once you get used to it.

### Limitations

The problem with using a to-do list is that most apps don't provide the functionality to only count “work days” when calculating repeat. Therefore, you might find you need to work on a project on weekends. Now, I don't really differentiate between weekdays and weekends. I do my personal hobby projects on weekends anyway, and most research work I do can be worked on at any time. Therefore, the to-do list app works just fine for me. However, this might be a problem for other people.

Another problem is that I can't temporarily change a repeating to-do to set off on another date in Things. This can happen if I need to temporarily work on a project on another day, but don't want to change the overall repeating pattern. If you use a to-do list app that doesn't support this, then there might be a problem.

### Future plans

This is why I want to develop an app specifically for Cicada's Method. I want it to be as simple as a to-do list, but with additional support to solve the problems above. I'm beginning to plan its roadmap and coming up with design ideas, and I'll post updates on my progress here.