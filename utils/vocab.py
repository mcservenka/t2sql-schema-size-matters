from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

EXCLUDED_LEXNAMES = {
    "noun.person",
    "noun.location",
    "noun.group",
}

# generated noun whitelist (by gpt-5)
COMMON_NOUN_WHITELIST  = [
    "ability", "access", "accident", "account", "achievement", "action", "activity",
    "advice", "affection", "agency", "agreement", "aid", "air", "airplane", "alarm",
    "alcohol", "ambition", "amount", "analysis", "angle", "animal", "answer", "anxiety",
    "apology", "appeal", "appearance", "apple", "application", "appointment", "area",
    "argument", "arm", "army", "arrival", "art", "article", "artist", "aspect", "assignment",
    "assistance", "assumption", "attack", "attempt", "attention", "attitude", "audience",
    "author", "authority", "autumn", "average", "baby", "back", "background", "bag",
    "balance", "ball", "band", "bank", "bar", "base", "basis", "basket", "bath", "battle",
    "beach", "bear", "beauty", "bed", "bedroom", "beer", "beginning", "behavior", "belief",
    "bell", "belt", "bench", "benefit", "berry", "bird", "birth", "birthday", "bit", "bite",
    "black", "blade", "blanket", "block", "blood", "blow", "board", "boat", "body", "bone",
    "bonus", "book", "boot", "border", "bottle", "bottom", "box", "boy", "brain", "branch",
    "bread", "break", "breakfast", "breath", "brick", "bridge", "brief", "brother",
    "brush", "bubble", "bucket", "budget", "building", "bullet", "bumper", "bunch", "burn",
    "bus", "bush", "business", "butter", "button", "buyer", "cable", "cake", "calculator",
    "calendar", "camera", "camp", "campaign", "can", "cancer", "candidate", "candle",
    "candy", "cap", "capacity", "capital", "car", "card", "care", "career", "carpet",
    "carrier", "case", "cash", "cat", "cause", "cell", "chain", "chair", "challenge",
    "chance", "change", "channel", "chapter", "character", "charge", "chart", "check",
    "cheek", "cheese", "chemical", "chest", "chicken", "child", "childhood", "chip",
    "choice", "church", "cigarette", "city", "class", "classic", "clerk", "client",
    "climate", "clock", "closet", "cloth", "clothes", "cloud", "club", "coach",
    "coal", "coast", "coat", "code", "coffee", "cold", "collar", "collection",
    "college", "color", "column", "combination", "comfort", "command", "comment",
    "commercial", "commission", "communication", "community", "company", "comparison",
    "competition", "complaint", "complex", "computer", "concept", "concern",
    "condition", "conference", "confidence", "conflict", "confusion", "connection",
    "consequence", "construction", "contact", "content", "contest", "context",
    "contract", "contribution", "control", "conversation", "cook", "cookie", "corner",
    "cost", "couch", "country", "courage", "course", "court", "cousin", "cover",
    "cow", "crack", "craft", "crash", "cream", "creation", "credit", "crew", "crime",
    "crisis", "critic", "crop", "cross", "crowd", "cry", "culture", "cup", "cupboard",
    "currency", "curve", "customer", "cycle", "dad", "damage", "dance", "danger",
    "data", "database", "daughter", "day", "deadline", "deal", "dealer", "death",
    "debate", "debt", "decision", "deck", "definition", "degree", "delivery", "demand",
    "department", "departure", "deposit", "description", "design", "desk", "detail",
    "development", "device", "diamond", "diet", "difference", "difficulty", "dig",
    "dimension", "dinner", "direction", "dirt", "disaster", "discipline", "discount",
    "discussion", "disease", "dish", "disk", "display", "distance", "distribution",
    "disturbance", "document", "dog", "door", "dot", "double", "doubt", "draft",
    "drama", "drawer", "drawing", "dream", "dress", "drink", "drive", "driver", "drop",
    "drug", "drum", "duck", "due", "dust", "duty", "ear", "earth", "ease", "east",
    "economy", "edge", "editor", "education", "effect", "effort", "egg", "election",
    "elevator", "emergency", "emotion", "emphasis", "employee", "employer", "end",
    "energy", "engine", "engineer", "entertainment", "enthusiasm", "entry", "environment",
    "equipment", "error", "escape", "essay", "establishment", "estate", "evening", "event",
    "example", "exchange", "excitement", "excuse", "exercise", "exit", "experience",
    "expert", "explanation", "expression", "extent", "eye", "fact", "factory", "failure",
    "faith", "fall", "family", "fan", "farm", "farmer", "fashion", "fat", "father",
    "fault", "fear", "feather", "feature", "fee", "feedback", "feeling", "female",
    "fiction", "field", "fight", "figure", "file", "film", "filter", "final", "finance",
    "finding", "finger", "finish", "fire", "fish", "fishing", "fit", "fix", "flag", "flash",
    "flight", "floor", "flower", "focus", "fold", "food", "foot", "football", "force",
    "forest", "fork", "form", "fortune", "foundation", "frame", "freedom", "frequency",
    "friend", "friendship", "front", "fruit", "fuel", "fun", "function", "fund", "future",
    "game", "gap", "garage", "garden", "gas", "gate", "gathering", "gear", "gene",
    "general", "generation", "genre", "gentleman", "gift", "girl", "glass", "glove",
    "goal", "god", "gold", "golf", "good", "government", "grain", "grandmother",
    "grass", "gravity", "greatness", "green", "ground", "group", "growth", "guarantee",
    "guard", "guest", "guide", "guitar", "habit", "hair", "half", "hall", "hand", "handle",
    "happiness", "harbor", "harm", "hat", "head", "health", "hearing", "heart",
    "heat", "height", "hell", "help", "hero", "hidden", "highlight", "hill", "hint",
    "hip", "history", "hobby", "hole", "holiday", "home", "honey", "hook", "hope",
    "horse", "hospital", "host", "hotel", "hour", "house", "housing", "human",
    "hunger", "hunt", "hurricane", "husband", "ice", "idea", "identity", "illness",
    "image", "impact", "importance", "impression", "improvement", "incident",
    "income", "increase", "independence", "industry", "infant", "inflation", "info",
    "information", "initiative", "injury", "inside", "insight", "inspection", "instance",
    "instruction", "insurance", "intention", "interaction", "interest", "interior",
    "internet", "interview", "introduction", "investment", "iron", "island", "issue",
    "item", "jacket", "job", "join", "joke", "journey", "joy", "judge", "juice", "jump",
    "junior", "jury", "justification", "key", "kick", "kid", "kill", "kind", "king",
    "kiss", "kitchen", "knee", "knife", "knowledge", "lab", "labor", "lack", "lady",
    "lake", "lamp", "land", "language", "law", "lawyer", "layer", "lead", "leader",
    "leaf", "league", "learning", "leather", "lecture", "leg", "legend", "length",
    "lesson", "letter", "level", "library", "life", "lift", "light", "limit", "line",
    "link", "list", "literature", "living", "load", "loan", "local", "location",
    "lock", "log", "loneliness", "look", "loss", "love", "low", "luck", "lunch",
    "machine", "magazine", "mail", "main", "maintenance", "major", "male",
    "mall", "man", "management", "manager", "manner", "manufacturer", "map",
    "market", "marriage", "mass", "match", "material", "math", "matter", "meal",
    "meaning", "measure", "meat", "media", "medicine", "medium", "meeting",
    "member", "memory", "menu", "message", "metal", "method", "middle", "midnight",
    "mind", "mine", "minimum", "minister", "minute", "mirror", "miss", "mission",
    "mistake", "mix", "mixture", "mode", "model", "moment", "money", "monitor",
    "month", "mood", "moon", "morning", "mortgage", "mother", "motion", "motor",
    "mountain", "mouse", "mouth", "move", "movie", "mud", "muscle", "music",
    "nail", "name", "nation", "nature", "need", "negotiation", "nerve", "network",
    "news", "night", "noise", "normal", "north", "note", "nothing", "notice",
    "number", "object", "obligation", "occasion", "office", "oil", "operation",
    "opinion", "opportunity", "opposite", "option", "orange", "order", "organization",
    "origin", "outcome", "outside", "oven", "owner", "pace", "pack", "package",
    "page", "pain", "paint", "painting", "pair", "pants", "paper", "parent", "park",
    "part", "participant", "partner", "party", "pass", "passage", "passenger",
    "passion", "past", "path", "patient", "pattern", "pause", "payment", "peace",
    "peak", "pen", "penalty", "pencil", "people", "percentage", "perception",
    "performance", "period", "permission", "person", "perspective", "phase",
    "phone", "photo", "piano", "picture", "piece", "pin", "pipe", "pitch", "place",
    "plan", "plane", "plant", "plastic", "plate", "platform", "player", "please",
    "pleasure", "plenty", "poem", "poet", "point", "police", "policy", "pollution",
    "pool", "pop", "population", "position", "positive", "possession", "possibility",
    "post", "pot", "potato", "poverty", "power", "practice", "preference", "preparation",
    "presence", "present", "pressure", "price", "pride", "priest", "primary",
    "principle", "print", "priority", "prison", "private", "probability", "problem",
    "procedure", "process", "produce", "product", "profession", "professor",
    "profit", "program", "progress", "project", "promise", "promotion", "proof",
    "property", "proposal", "protection", "psychology", "public", "pull", "punishment",
    "purchase", "purpose", "push", "quality", "quantity", "quarter", "queen",
    "question", "quiet", "quote", "race", "radio", "rain", "range", "rate", "ratio",
    "reaction", "reader", "reading", "reason", "reality", "receipt", "recipe",
    "recognition", "record", "recording", "recovery", "reflection", "refrigerator",
    "region", "register", "regret", "relation", "relationship", "relative", "release",
    "relief", "religion", "rent", "repair", "repeat", "replacement", "reply", "report",
    "representative", "republic", "request", "research", "reserve", "resistance",
    "resolution", "resource", "response", "responsibility", "restaurant", "result",
    "return", "revenue", "review", "reward", "rice", "ride", "ring", "risk", "river",
    "road", "rock", "role", "roof", "room", "rope", "round", "route", "routine",
    "row", "rubber", "rule", "run", "sadness", "safety", "sail", "sailor", "salad",
    "salary", "sale", "salt", "sample", "sand", "satisfaction", "scale", "scene",
    "schedule", "scheme", "school", "science", "screen", "screw", "sea", "search",
    "season", "seat", "second", "secret", "section", "sector", "security", "selection",
    "self", "sense", "sentence", "series", "service", "session", "setting", "sex",
    "shade", "shadow", "shake", "shame", "shape", "share", "shelter", "shift",
    "shine", "ship", "shirt", "shock", "shoe", "shop", "shopping", "shot", "shoulder",
    "show", "shower", "sick", "side", "sign", "signal", "signature", "silence",
    "silver", "simple", "simplicity", "sing", "singer", "sister", "site", "situation",
    "size", "skill", "skin", "sky", "sleep", "slope", "smell", "smile", "smoke",
    "snow", "society", "sock", "softness", "soil", "solution", "song", "sound",
    "soup", "source", "space", "speaker", "specialist", "species", "speech",
    "speed", "spirit", "spite", "split", "sport", "spot", "spray", "spring", "square",
    "stable", "staff", "stage", "stair", "stake", "stall", "stamp", "stand", "standard",
    "star", "start", "statement", "station", "statistics", "status", "stay",
    "steak", "steel", "step", "stick", "stock", "stomach", "stone", "stop",
    "storage", "store", "storm", "story", "strain", "strategy", "stream", "street",
    "strength", "stress", "structure", "student", "studio", "study", "stuff", "style",
    "subject", "substance", "success", "suggestion", "summer", "sun", "support",
    "surface", "surprise", "survival", "suspicion", "swim", "swing", "switch",
    "system", "table", "tackle", "tale", "talk", "tank", "tap", "target", "task",
    "taste", "tax", "tea", "teacher", "teaching", "team", "tear", "technology",
    "teen", "telephone", "television", "temperature", "tend", "tennis", "tension",
    "term", "test", "text", "thanks", "theory", "thing", "thought", "thread",
    "throat", "ticket", "tie", "time", "tire", "title", "toast", "tobacco", "today",
    "toe", "tone", "tongue", "tool", "tooth", "topic", "total", "touch", "tour",
    "towel", "tower", "town", "toy", "track", "trade", "tradition", "traffic",
    "train", "trainer", "training", "transaction", "transition", "transport",
    "trash", "travel", "treat", "tree", "trial", "trick", "trouble", "truck",
    "trust", "truth", "try", "tube", "tune", "turn", "type", "uncle", "understanding",
    "union", "unique", "unit", "universe", "unknown", "upper", "use", "user",
    "valley", "value", "variety", "vehicle", "version", "vessel", "video", "view",
    "virtue", "voice", "volume", "wait", "walk", "wall", "war", "warning", "wash",
    "waste", "water", "wave", "way", "wealth", "weather", "wedding", "week",
    "weekend", "weight", "welcome", "west", "wheel", "while", "whisper",
    "white", "whole", "wife", "will", "wind", "window", "wine", "winter",
    "wire", "wish", "woman", "wood", "word", "work", "worker", "world",
    "worry", "worth", "wrap", "writer", "writing", "yard", "year", "youth",
    "zone"
]

# load clean noun vocab
def load_noun_vocabulary(min_len=4):
    nouns = set()
    for syn in wn.all_synsets(pos=wn.NOUN):
        
        if syn.lexname() in EXCLUDED_LEXNAMES:
            continue
        
        for lemma in syn.lemma_names():
            
            lemma = lemma.lower()

            if not lemma.isalpha():
                continue
            if "_" in lemma:
                continue
            if len(lemma) < min_len:
                continue
            if has_excluded_sense(lemma):
                continue
            
            z = zipf_frequency(lemma, "en")

            # use zipfto get proper and common nouns
            if z > 4.0 and lemma not in COMMON_NOUN_WHITELIST:
                continue

            nouns.add(lemma)

    return nouns

def has_excluded_sense(word: str) -> bool:
    synsets = wn.synsets(word, pos=wn.NOUN)
    for syn in synsets:
        if syn.lexname() in EXCLUDED_LEXNAMES:
            return True
    return False

