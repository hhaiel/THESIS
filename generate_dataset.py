import pandas as pd


# Create a list of troll comments for testing
# Mix of English, Tagalog, and Taglish with varying levels of toxicity

troll_comments = [
    # English troll comments - high toxicity
    "THIS IS THE WORST VIDEO EVER MADE!!! 🤬🤬🤬",
    "you're so stupid I can't even believe it 💀💀💀",
    "NOBODY CARES ABOUT YOUR OPINION!!!!! 🤡🤡🤡",
    "delete your account please you're embarrassing yourself",
    "this is absolute GARBAGE content unfollow immediately 🗑️",
    "how are you even allowed to post this trash? 🤮",
    "CANCELLED!!! This person is problematic AF",
    "your content is so cringe it physically hurts me",
    "what a LOSER!!! hahahaha 🤣🤣🤣",
    "do everyone a favor and STOP POSTING",
    
    # English troll comments - medium toxicity
    "mid content as usual 🥱",
    "this flopped hard lol 😂",
    "didn't ask + ratio + you fell off",
    "respect my opinion: you have zero talent",
    "basic content for basic people I guess",
    "imagine thinking this is good content 🙄",
    "BORING!!! make better content please",
    "you clearly don't know what you're talking about",
    "your followers must be bots because this is terrible",
    "yikes... this ain't it chief 😬",
    
    # Filipino troll comments - high toxicity
    "ANG BOBO MO TALAGA!!! HAHAHAHAHA 🤡🤡🤡",
    "TANGA NAMAN NITO HINDI ALAM GINAGAWA 💀💀💀",
    "putangina mo talaga napaka walang kwenta",
    "GAGO KA BA??? SOBRANG PANGET NG CONTENT MO",
    "bobo amputa di mo ba alam ginagawa mo???",
    "ang pangit ng content wag na mag upload please",
    "KADIRI TALAGA TONG TAO NA TO 🤮🤮🤮",
    "BWISIT NAMAN OO!!! nakaka inis ka talaga",
    "BASURA CONTENT!!! wag na mag post",
    "AMPANGET MO TALAGA KADIRI 🤮🤮🤮",

    # Filipino troll comments - medium toxicity
    "naka umay naman to 🙄",
    "wala talagang kwenta tong content na to",
    "loko loko talaga tong taong to eh",
    "dami satsat wala namang laman",
    "respect my opinion pero pangit talaga to",
    "sino bang nanonood sa ganitong content? walang kwenta",
    "parang tanga lang? bat ka ganyan?",
    "UMAY!!! paulit ulit lang content mo",
    "wala kang alam sa ginagawa mo 🙄",
    "kalat lang talaga content mo",
    
    # Taglish troll comments - high toxicity
    "HAHAHA BOBO NAMAN NITO!!! you don't know anything 🤡🤡🤡",
    "you're so TANGA!!! delete this video now na please!!!",
    "ang PANGIT ng content mo!!! please lang wag na mag post",
    "GAGO KA BA? this is the worst content I've seen!!!",
    "your face is so PANGET hahaha kadiri talaga 🤮",
    "you need to STOP NA!!! nakakahiya ka",
    "tang*na this is trash talaga!!! nakakabwisit ka",
    "I HATE YOUR CONTENT pangit talaga 🗑️🗑️🗑️",
    "BOBO NAMAN NITO!!! you clearly don't know what you're doing",
    "pisting yawa CRINGE content!!!! 🤮🤮🤮",
    
    # Taglish troll comments - medium toxicity
    "your content is so BASURA talaga",
    "wag na please this is embarrassing yourself",
    "UMAY sa content mo its always the same",
    "how to unsee, napaka sama ng content mo",
    "respect my opinion na lang but this is PANGET",
    "bakit ganito content mo? so disappointing",
    "cringe talaga you should just stop",
    "try to make better content naman please",
    "nobody asked for this KADIRI content",
    "YIKES talaga your content is so bad 😬",
    
    # Filipino political troll comments
    "dilawan ka siguro kaya ganyan ka mag-isip",
    "PINKLAWAN SPOTTED!!! 🤣🤣🤣",
    "bayaran ka ba ng mga aquino? halata naman",
    "NPA supporter spotted delete this!!!",
    "puppet ka lang ng mga komunista 🤡",
    "lutang supporter ka talaga no? 🤣",
    "DIEHARD BBM KAMI DEAL WITH IT!!!",
    "DELAWAN PROPAGANDA!!!!! 🤮🤮🤮",
    "kitang kita naman na bayaran ka ng oligarchs",
    "respeto na lang po sa presidente!!! HATER!!!",
    
    # Spam/scam troll comments
    "EARN 50,000 PESOS DAILY!!! Click my profile NOW!!",
    "Check my bio for easy money!!! 💰💰💰",
    "FREE LOAD JUST TEXT 09123456789 NOW!!!",
    "get rich quick!!! message me NOW!! 💸💸💸",
    "I MADE ₱100,000 FROM THIS!!! CHECK MY BIO!!",
    "FOREX TRADING EASY MONEY!!! PM ME 🤑🤑🤑",
    "Salamat sa ₱50,000 daily income!!! Click link now!!",
    "SANA ALL EARNING 6 DIGITS!!! Click my profile to know how",
    "Bili na ng PESO FARM ACCOUNT!!! PM me!!!",
    "LOOKING FOR GCASH CASHOUT!!! PM ME NOW!!!",
]

# Create a list of non-troll comments for contrast
non_troll_comments = [
    # English positive comments
    "Great video! Thanks for the information.",
    "I really enjoyed your content, very helpful!",
    "This is exactly what I needed to see today.",
    "You explained this so well! Thank you.",
    "Love your editing style, very professional.",
    "This is such quality content, keep it up!",
    "I've learned so much from your videos, thank you!",
    "Your voice is so soothing, love the narration.",
    "This is the best explanation I've seen so far.",
    "Very informative and well-researched!",
    
    # English neutral comments
    "Interesting perspective, I'll think about this.",
    "First time watching your content, nice to discover you.",
    "I'm not sure I agree, but I respect your opinion.",
    "Watching this during my lunch break.",
    "Does anyone know the song at 2:15?",
    "I just found this channel today.",
    "I've been following this topic for a while.",
    "Interesting, would like to see more on this.",
    "This reminds me of something I saw last week.",
    "Here before this blows up.",
    
    # Filipino positive comments
    "Ang ganda ng video mo! Maraming salamat.",
    "Sobrang nakakatuwa yung content mo.",
    "Ang galing mo mag-explain, salamat!",
    "Napaka informative nito, salamat sa pag share.",
    "Ang husay ng presentation, good job!",
    "Sana all ganito kagaling mag explain.",
    "Idol talaga kita, ang galing mo!",
    "Salamat sa info, napakalaking tulong.",
    "Solid content talaga, pabor ito!",
    "Galing mo talaga, keep it up!",
    
    # Filipino neutral comments
    "Nakita ko lang to sa FYP ko.",
    "First time ko to napanood, interesting.",
    "Nag scroll lang ako tapos nakita ko to.",
    "Napaisip ako dito ah, interesting.",
    "May point ka naman, pero di ko sure kung agree ako.",
    "Medyo confused pa ako pero interesting.",
    "Ito ba yung trending ngayon?",
    "Sino dito nanonood habang kumakain?",
    "Di ko pa napapanood yung buong series.",
    "Bago lang ako dito sa channel mo.",
    
    # Taglish positive comments
    "Your content is so galing talaga! Thank you!",
    "Ang informative nito, thanks for sharing!",
    "I always look forward to your videos kasi ang galing mo!",
    "This is so helpful talaga, salamat!",
    "Napaka interesting ng topic na to, good job!",
    "Keep up the good work ha, nakakatuwa yung videos mo.",
    "Your energy is so nakakahawa, love watching!",
    "Thank you sa information, very useful!",
    "Ang ganda ng explanation mo, I understand it na!",
    "Super helpful talaga, salamat so much!",
    
    # Taglish neutral comments
    "Just watching this habang nagkakape.",
    "I'm curious kung ano next topic mo.",
    "Medyo similar ito sa nakita ko before pero okay pa rin.",
    "First time ko to napanood, interesting siya.",
    "I wonder kung meron ka pang ibang topics like this.",
    "Napaisip ako about this topic, interesting siya.",
    "Sino dito fan from day one? Just curious.",
    "Bago lang ako dito sa channel mo, nice content pala.",
    "May point ka naman, but I need to research pa.",
    "Interesting yung take mo on this topic."
]

# Create a balanced dataset with both troll and non-troll comments
all_comments = []
all_labels = []

# Add troll comments
for comment in troll_comments:
    all_comments.append(comment)
    all_labels.append(1)  # 1 indicates troll comment

# Add non-troll comments
for comment in non_troll_comments:
    all_comments.append(comment)
    all_labels.append(0)  # 0 indicates non-troll comment

# Create DataFrame
df = pd.DataFrame({
    'comment': all_comments,
    'is_troll': all_labels
})

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('troll_comments_dataset.csv', index=False)

print(f"Dataset created with {len(df)} comments ({sum(all_labels)} troll comments, {len(all_labels) - sum(all_labels)} non-troll comments)")
print("Saved as 'troll_comments_dataset.csv'")

# Display a few examples
print("\nExample troll comments:")
print(df[df['is_troll'] == 1]['comment'].head(5).to_string())

print("\nExample non-troll comments:")
print(df[df['is_troll'] == 0]['comment'].head(5).to_string())