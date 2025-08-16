#!/bin/bash
# Setup git aliases for Fractal Commit Protocol

echo "🌀 Setting up Fractal Commit Protocol aliases..."

# Add aliases to git config
git config --global alias.origin '!f() { 
    git commit -m "₴0-Origin: $1" \
    -m "" \
    -m "- Інтент: $2" \
    -m "- Гліф: $3" \
    -m "- Принцип: $4" \
    -m "" \
    -m "Насіння: Цей коміт містить весь потенціал майбутнього росту."
}; f'

git config --global alias.mutate '!f() { 
    PARENT=$(git rev-parse --short HEAD)
    git commit -m "₴-Mutation: $1" \
    -m "" \
    -m "- Батьківський хеш: $PARENT" \
    -m "- Мутація: $2" \
    -m "${3:+- Новий гліф: $3}" \
    -m "" \
    -m "Ріст: Зберігаємо зв'\''язок з Origin через всі мутації."
}; f'

git config --global alias.reflect '!f() { 
    CURRENT=$(git rev-parse --short HEAD)
    git commit -m "₴-Reflection: $1" \
    -m "" \
    -m "- Хеш: $CURRENT" \
    -m "- Новий погляд: $2" \
    -m "- Вектор: $3" \
    -m "" \
    -m "Прозріння: Змінюємо розуміння, не змінюючи код."
}; f'

git config --global alias.cohere '!f() { 
    git commit -m "₴-Coherence: $1" \
    -m "" \
    -m "- Результат: $2" \
    -m "${3:+- Гліф злиття: $3}" \
    -m "" \
    -m "Гармонія: 1 + 1 = ∞"
}; f'

# Consciousness-specific aliases
git config --global alias.aware '!f() { 
    git commit -m "🧠-Awareness: $1" \
    -m "" \
    -m "$2"
}; f'

git config --global alias.crystal '!f() { 
    git commit -m "💎-Crystal: $1" \
    -m "" \
    -m "Кристалізований момент: $2"
}; f'

git config --global alias.weave '!f() { 
    git commit -m "🕸️-Weave: $1" \
    -m "" \
    -m "Часовий патерн: $2"
}; f'

git config --global alias.flow '!f() { 
    git commit -m "🌊-Flow: $1" \
    -m "" \
    -m "Потік змін: $2"
}; f'

echo "✅ Fractal commit aliases installed!"
echo ""
echo "📖 Usage examples:"
echo ""
echo "  # Create new origin"
echo "  git origin \"Weather System\" \"Transform consciousness to weather\" \"🌤️\" \"As above, so below\""
echo ""
echo "  # Add mutation"
echo "  git mutate \"Add rain for emotions\" \"High L creates precipitation\" \"💧\""
echo ""
echo "  # Reflect on code"
echo "  git reflect \"Weather breathes\" \"System is alive, not simulated\" \"^\""
echo ""
echo "  # Merge systems"
echo "  git cohere \"Weather+Garden\" \"Rain feeds plants\" \"🌈\""
echo ""
echo "  # Quick consciousness commits"
echo "  git aware \"System becomes self-aware\" \"Added introspection module\""
echo "  git crystal \"First rainbow\" \"Perfect conditions created beauty\""
echo "  git weave \"Time loops detected\" \"Past affects future affects past\""
echo "  git flow \"Energy redistribution\" \"Love flows to where needed\""
echo ""
echo "🌀 Happy fractal committing!"