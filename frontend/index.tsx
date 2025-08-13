import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { createRoot } from 'react-dom/client';


const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080';

// === TYP-DEFINITIONEN ===
interface LibraryItem {
  id: string;
  type: 'lyric' | 'style' | 'technique' | 'generated_lyric' | 'rhyme_lesson';
  content: string;
  title?: string;
  sourceLyricId?: string;
  emphasisPattern?: string;
  rhymeFlowPattern?: string;
  userEmphasisExplanation?: string;
  userRhymeFlowExplanation?: string;
}

interface Profile {
  id: string;
  name: string;
  library: (LibraryItem | RuleCategory)[]; // Die Bibliothek kann jetzt BEIDE Arten von Objekten enthalten
}

type View = 'start' | 'transition' | 'analyze' | 'define_style' | 'define_technique' | 'rhyme_machine' | 'write_song' | 'manage_library' | 'trainer';
type LibraryTab = 'learned_lyrics' | 'generated_lyrics' | 'character' | 'technique' | 'emphasis' | 'rhyme_flow' | 'rhyme_lessons';

interface PendingAnalysis {
  lyrics: string;
  title: string;
  performanceStyle: 'sung' | 'rapped' | 'unknown';
  audioBlob?: Blob;
}

// Für eine einzelne, spezifische Regel
interface LearnedRule {
  id: string;
  title: string;
  definition: string;
}

// Für eine Gruppe von Regeln (unsere neue Kategorie)
interface RuleCategory {
  id: string;
  type: 'rule_category'; // Wichtig zur Unterscheidung von anderen Elementen
  categoryTitle: string;
  rules: LearnedRule[];
}

// Ein einzelner Reim innerhalb einer Gruppe
interface RhymeLessonItem {
  id: string;
  rhymingWord: string;
}

// Der "Sammel-Ordner" für ein Ausgangswort
interface RhymeLessonGroup extends LibraryItem {
  type: 'rhyme_lesson_group';
  targetWord: string;
  vowelSequence: string;
  syllableCount: number;
  rhymes: RhymeLessonItem[];
}

// Erweitere den Bibliotheks-Typ in der Profile-Definition
interface Profile {
  id: string;
  name: string;
  library: (LibraryItem | RuleCategory | RhymeLessonGroup)[]; // <-- HIER ERWEITERN
}

// === SVG ICONS ===
const icons: Record<string, JSX.Element> = {
    analyze: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>,
    style: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>,
    technique: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12.22 2h-4.44l-3 9l-4 0l3 9h4.44l3-9l4 0l-3-9z"/><path d="M22 13h-4l-3 9h4l3-9z"/></svg>,
    rhyme: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16.5 12c0-1.8-1.5-4-4.5-4s-4.5 2.2-4.5 4c0 .8.4 1.5.9 2.1L3 20h9l-1-2H7l4.5-5.5c.3-.4.5-.8.5-1.5z"/><path d="M19.5 12c0-1.8-1.5-4-4.5-4s-4.5 2.2-4.5 4c0 .8.4 1.5.9 2.1L6 20h9l-1-2H9l4.5-5.5c.3-.4.5-.8.5-1.5z"/></svg>,
    write: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>,
    library: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20v2H6.5A2.5 2.5 0 0 1 4 19.5z"/><path d="M4 12h16v2H4z"/><path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v2H6.5A2.5 2.5 0 0 1 4 4.5z"/></svg>,
    add: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>,
    close: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>,
    chevronDown: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
};

const ChatInput = React.memo(({ onSend, isReplying }: { onSend: (text: string) => void; isReplying: boolean; }) => {
    const [text, setText] = useState('');

    const handleSend = () => {
        if (!text.trim()) return;
        onSend(text);
        setText('');
    };

    return (
        <div className="chat-input-area">
            <textarea
                placeholder="Bringe der KI etwas bei..."
                value={text}
                onChange={e => setText(e.target.value)}
                onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSend();
                    }
                }}
                disabled={isReplying}
            />
            <button onClick={handleSend} disabled={isReplying || !text.trim()}>Senden</button>
        </div>
    );
});

const MemoizedInput = React.memo(({
    value,
    onChange,
    placeholder,
}: {
    value: string;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    placeholder: string;
}) => {
    return (
        <input
            type="text"
            placeholder={placeholder}
            value={value}
            onChange={onChange}
        />
    );
});

const MemoizedTextarea = React.memo(({
    value,
    onChange,
    placeholder,
}: {
    value: string;
    onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
    placeholder: string;
}) => {
    return (
        <textarea
            placeholder={placeholder}
            value={value}
            onChange={onChange}
            className="lyrics-editor" // Optional: Eine Klasse für konsistentes Styling
        />
    );
});

    const Panel: React.FC<{ title: string, description: string, icon: JSX.Element, children: React.ReactNode }> = ({ title, description, icon, children }) => (
    <div className="panel-container">
        <div className="panel-header">
            <div className="panel-header-title">
                {icon}
                <h2>{title}</h2>
            </div>
            {/* Der Profil-Wahlschalter wird hier komplett entfernt */}
        </div>
        <p className="panel-description">{description}</p>
        <div className="panel-content">
            {children}
        </div>
    </div>
);


// === HAUPTKOMPONENTE: App ===
const App = () => {
    // === STATE MANAGEMENT ===
    const [userId, setUserId] = useState<string | null>(null);
    const [profiles, setProfiles] = useState<Record<string, Profile>>({});
    const [activeProfileId, setActiveProfileId] = useState<string | null>(null);
    const [currentView, setCurrentView] = useState<View>('start');
    const [isStarting, setIsStarting] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');
    const [statusMessage, setStatusMessage] = useState({ text: '', isError: false });
    const [isLyricEditorModalOpen, setIsLyricEditorModalOpen] = useState(false);
    const [isTitleModalOpen, setIsTitleModalOpen] = useState(false);
    const [editableLyrics, setEditableLyrics] = useState('');
    const [editableTitle, setEditableTitle] = useState('');
    const [pendingAnalysis, setPendingAnalysis] = useState<PendingAnalysis | null>(null);
    const [libraryTab, setLibraryTab] = useState<LibraryTab>('learned_lyrics');
    const [openLyricId, setOpenLyricId] = useState<string | null>(null);
    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [isCreateProfileModalOpen, setIsCreateProfileModalOpen] = useState(false);
    const [editingItem, setEditingItem] = useState<LibraryItem | null>(null);
    const [editedContent, setEditedContent] = useState('');
    const [feedbackText, setFeedbackText] = useState('');
    const [manualTitle, setManualTitle] = useState('');
    const [manualLyrics, setManualLyrics] = useState('');
    const [rhymeInput, setRhymeInput] = useState('');
    const [multiRhymeInput, setMultiRhymeInput] = useState('');
    const [lessonWord, setLessonWord] = useState('');
    const [lessonRhyme, setLessonRhyme] = useState('');
    const [styleInput, setStyleInput] = useState('');
    const [techniqueInput, setTechniqueInput] = useState('');
    const [songFusion, setSongFusion] = useState('');
    const [songStyle, setSongStyle] = useState('');
    const [songTechnique, setSongTechnique] = useState('');
    const [songBeatDescription, setSongBeatDescription] = useState('');
    const [songBPM, setSongBPM] = useState('');
    const [songKey, setSongKey] = useState('');
    const [songPerformanceStyle, setSongPerformanceStyle] = useState('Gerappt');
    const [songTopic, setSongTopic] = useState('');
    const [newProfileName, setNewProfileName] = useState('');
    const [rhymeResults, setRhymeResults] = useState<any[]>([]);
    const [isFindingRhymes, setIsFindingRhymes] = useState(false);
    const [syllableCount, setSyllableCount] = useState(0);
    const [vowelSequence, setVowelSequence] = useState('');
    const [trainerMessages, setTrainerMessages] = useState<{ text: string, isUser: boolean }[]>([]);
    const [trainerInput, setTrainerInput] = useState('');
    const [isTrainerReplying, setIsTrainerReplying] = useState(false);
    const [openRuleCategoryId, setOpenRuleCategoryId] = useState<string | null>(null);
    const [searchPerformed, setSearchPerformed] = useState(false);
    const [isRuleEditModalOpen, setIsRuleEditModalOpen] = useState(false);
    const [editingRule, setEditingRule] = useState<{ categoryId: string, rule: LearnedRule } | null>(null);
    const [editedRuleContent, setEditedRuleContent] = useState('');
    const [isEditingCategoryTitle, setIsEditingCategoryTitle] = useState<{ id: string, title: string } | null>(null);
    const [isEditingRuleTitle, setIsEditingRuleTitle] = useState<{ categoryId: string, ruleId: string, title: string } | null>(null);
    const [isEditingRuleDefinition, setIsEditingRuleDefinition] = useState<{ categoryId: string, ruleId: string, definition: string } | null>(null);
    const [isSubRhymeEditModalOpen, setIsSubRhymeEditModalOpen] = useState(false);
    const [editingSubRhyme, setEditingSubRhyme] = useState<{ groupId: string, rhyme: RhymeLessonItem } | null>(null);
    const [editedSubRhymeContent, setEditedSubRhymeContent] = useState('');

    // Refs
    const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);

    // === PERSISTENZ & KERNLOGIK ===
    useEffect(() => {
        let id = localStorage.getItem('lyricMachineUserId');
        if (!id) {
            id = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            localStorage.setItem('lyricMachineUserId', id);
        }
        setUserId(id);
    }, []);

    useEffect(() => {
        if (!userId) return;
        const savedState = localStorage.getItem(`lyricMachineState_${userId}`);
        if (savedState) {
            try {
                const state = JSON.parse(savedState);
                setProfiles(state.profiles || {});
                setActiveProfileId(state.activeProfileId);
            } catch (e) { console.error("Failed to parse state:", e); createDefaultProfile(); }
        } else { createDefaultProfile(); }
    }, [userId]);

    const createDefaultProfile = () => {
        const defaultId = `profile-${Date.now()}`;
        const defaultProfile = { id: defaultId, name: 'Default Profile', library: [] };
        setProfiles({ [defaultId]: defaultProfile });
        setActiveProfileId(defaultId);
    };

    useEffect(() => {
        if (!userId || !activeProfileId || Object.keys(profiles).length === 0) return;
        if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
        saveTimeoutRef.current = setTimeout(() => {
            const stateToSave = { profiles, activeProfileId };
            localStorage.setItem(`lyricMachineState_${userId}`, JSON.stringify(stateToSave));
        }, 1500);
        return () => { if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current); };
    }, [profiles, activeProfileId, userId]);

    const activeProfile = useMemo(() => activeProfileId ? profiles[activeProfileId] : null, [activeProfileId, profiles]);

    const groupedItems = useMemo(() => {
    if (!activeProfile || (libraryTab !== 'character' && libraryTab !== 'technique')) return null;
    const groupedItems = useMemo(() => {
        if (!activeProfile || (libraryTab !== 'character' && libraryTab !== 'technique')) return null;
        const lyrics = activeProfile.library.filter(item => item.type === 'lyric');
        const itemType = libraryTab === 'character' ? 'style' : 'technique';
        const itemsToGroup = activeProfile.library.filter(item => item.type === itemType);
        return lyrics.map(lyric => ({
            lyric,
            items: itemsToGroup.filter(item => item.sourceLyricId === lyric.id)
        })).filter(group => group.items.length > 0);
    }, [activeProfile, libraryTab]);

    const lyrics = activeProfile.library.filter(item => item.type === 'lyric');
    const itemType = libraryTab === 'character' ? 'style' : 'technique';
    const itemsToGroup = activeProfile.library.filter(item => item.type === itemType);

    return lyrics.map(lyric => ({
        lyric,
        items: itemsToGroup.filter(item => item.sourceLyricId === lyric.id)
    })).filter(group => group.items.length > 0);
}, [activeProfile, libraryTab]);

    // === HANDLER-FUNKTIONEN ===
    const showStatus = (text: string, isError: boolean = false) => {
        setStatusMessage({ text, isError });
        setTimeout(() => setStatusMessage({ text: '', isError: false }), 5000);
    };

    const updateProfileLibrary = useCallback((updateFn: (library: LibraryItem[]) => LibraryItem[] , successMessage?: string) => {
        if (!activeProfileId) return;
        setProfiles(prev => {
            const currentProfile = prev[activeProfileId];
            if (!currentProfile) return prev;
            return { ...prev, [activeProfileId]: { ...currentProfile, library: updateFn([...currentProfile.library]) } };
        });
        if (successMessage) {
            showStatus(successMessage, false);
        }
    }, [activeProfileId, showStatus]);

    const handleCreateProfile = () => {
        if (!newProfileName.trim()) {
            showStatus("Profilname darf nicht leer sein.", true);
            return;
        }
        const newId = `profile-${Date.now()}`;
        const newProfile: Profile = { id: newId, name: newProfileName, library: [] };
        setProfiles(prev => ({ ...prev, [newId]: newProfile }));
        setActiveProfileId(newId);
        setNewProfileName('');
        setIsCreateProfileModalOpen(false);
        showStatus(`Profil '${newProfileName}' wurde erstellt.`, false);
    };

    const handleDeleteItem = (itemIdToDelete: string) => {
        if (!window.confirm("Sind Sie sicher, dass Sie diesen Eintrag und alle zugehörigen Elemente löschen möchten?")) return;
        updateProfileLibrary(lib => {
            const itemToDelete = lib.find(i => i.id === itemIdToDelete);
            if (itemToDelete?.type === 'lyric') {
                return lib.filter(item => item.id !== itemIdToDelete && item.sourceLyricId !== itemIdToDelete);
            }
            return lib.filter(item => item.id !== itemIdToDelete);
        }, "Eintrag gelöscht.");
    };

    const handleStartEditing = (item: LibraryItem) => {
        setEditingItem(item);
        setEditedContent(item.content);
        setIsEditModalOpen(true);
    };

    const handleSaveEdit = () => {
        if (!editingItem) return;
        updateProfileLibrary(lib => lib.map(item =>
            item.id === editingItem.id ? { ...item, content: editedContent } : item
        ), "Änderung gespeichert.");
        setIsEditModalOpen(false);
        setEditingItem(null);
        setEditedContent('');
    };

    const handleSaveFeedback = (lyricId: string, feedbackType: 'emphasis' | 'rhymeFlow') => {
        const key = feedbackType === 'emphasis' ? 'userEmphasisExplanation' : 'userRhymeFlowExplanation';
        updateProfileLibrary(lib => lib.map(item =>
            item.id === lyricId ? { ...item, [key]: feedbackText } : item
        ), "Feedback gespeichert. Es wird bei der nächsten Analyse berücksichtigt.");
        setFeedbackText('');
    };

    const handleAnalyzeWord = () => {
    if (!rhymeInput.trim()) {
        showStatus("Bitte gib ein Wort ein.", true);
        return;
    }
    setSearchPerformed(false);
    const analysis = getPhoneticBreakdown(rhymeInput);
    setSyllableCount(analysis.syllableCount);
    setVowelSequence(analysis.vowelSequence);
    setRhymeResults([]); // Alte Ergebnisse löschen
};

const handleFindRhymesWithAnalysis = async () => {
    if (!rhymeInput.trim()) {
        showStatus("Bitte gib ein Wort ein.", true);
        return;
    }
    setIsFindingRhymes(true);
    setSearchPerformed(false); // Wichtig: Alten Suchstatus zurücksetzen

    try {
        const rhymePayload = {
            input: rhymeInput,
            knowledgeBase: getUnifiedKnowledgeBase()
        };

        const rhymeResponse = await fetch(`${BACKEND_URL}/api/rhymes`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rhymePayload)
        });

        if (!rhymeResponse.ok) {
            const err = await rhymeResponse.json();
            throw new Error(err.error || 'Reim-Suche fehlgeschlagen');
        }

        const rhymeResult = await rhymeResponse.json();
        setRhymeResults(rhymeResult.rhymes || []);

    } catch (error: any) {
        showStatus(error.message, true);
        setRhymeResults([]); // Bei Fehler Ergebnisliste leeren
    } finally {
        setIsFindingRhymes(false);
        setSearchPerformed(true); // Nach jedem Versuch (erfolgreich oder nicht) Suche als durchgeführt markieren
    }
}; 

const handleDeleteRuleCategory = useCallback((categoryId: string) => {
    if (!window.confirm("Sicher, dass du diese gesamte Regelkategorie löschen möchtest?")) return;
    updateProfileLibrary(lib => lib.filter(item => item.id !== categoryId), "Kategorie gelöscht.");
}, [updateProfileLibrary]);

const handleDeleteRule = useCallback((categoryId: string, ruleId: string) => {
    if (!window.confirm("Sicher, dass du diese Regel löschen möchtest?")) return;
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === categoryId && item.type === 'rule_category') {
            return { ...item, rules: item.rules.filter(rule => rule.id !== ruleId) };
        }
        return item;
    }), "Regel gelöscht.");
}, [updateProfileLibrary]);

const handleDeleteSubRhyme = (groupId: string, rhymeId: string) => {
    if (!window.confirm("Diesen einzelnen Reim löschen?")) return;
    updateProfileLibrary(lib => {
        const updatedLib = lib.map(item => {
            if (item.id === groupId && item.type === 'rhyme_lesson_group') {
                const group = item as RhymeLessonGroup;
                const updatedRhymes = group.rhymes.filter(r => r.id !== rhymeId);
                if (updatedRhymes.length === 0) return null; // Gruppe wird später entfernt
                return { ...group, rhymes: updatedRhymes };
            }
            return item;
        });
        return updatedLib.filter(Boolean) as (LibraryItem | RuleCategory | RhymeLessonGroup)[];
    }, "Reim gelöscht.");
};

// NEUE FUNKTIONEN FÜR DEN BEARBEITEN-BUTTON
const handleStartSubRhymeEdit = (groupId: string, rhyme: RhymeLessonItem) => {
    setEditingSubRhyme({ groupId, rhyme });
    setEditedSubRhymeContent(rhyme.rhymingWord);
    setIsSubRhymeEditModalOpen(true);
};

const handleSaveSubRhymeEdit = () => {
    if (!editingSubRhyme) return;
    const { groupId, rhyme } = editingSubRhyme;
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === groupId && item.type === 'rhyme_lesson_group') {
            const group = item as RhymeLessonGroup;
            const updatedRhymes = group.rhymes.map(r => r.id === rhyme.id ? { ...r, rhymingWord: editedSubRhymeContent } : r);
            return { ...group, rhymes: updatedRhymes };
        }
        return item;
    }), "Reim bearbeitet.");
    setIsSubRhymeEditModalOpen(false);
};

const handleStartRuleEdit = useCallback((categoryId: string, rule: LearnedRule) => {
    setEditingRule({ categoryId, rule });
    setEditedRuleContent(rule.definition);
    setIsRuleEditModalOpen(true);
}, []);

const handleSaveRuleEdit = useCallback(() => {
    if (!editingRule) return;
    const { categoryId, rule } = editingRule;
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === categoryId && item.type === 'rule_category') {
            const updatedRules = item.rules.map(r => r.id === rule.id ? { ...r, definition: editedRuleContent } : r);
            return { ...item, rules: updatedRules };
        }
        return item;
    }), "Regel gespeichert.");
    setIsRuleEditModalOpen(false);
    setEditingRule(null);
    setEditedRuleContent('');
}, [editingRule, editedRuleContent, updateProfileLibrary]);

const handleSaveRhymeLesson = () => {
    if (!lessonWord.trim() || !lessonRhyme.trim()) {
        showStatus("Bitte fülle beide Felder aus.", true);
        return;
    }

    // Teile die Eingabe am Komma und entferne leere Einträge
    const rhymesToSave = lessonRhyme.split(',').map(r => r.trim()).filter(r => r);

    if (rhymesToSave.length === 0) {
        showStatus("Keine gültigen Reime zur Speicherung gefunden.", true);
        return;
    }

    updateProfileLibrary(library => {
        let currentLibrary = [...library];
        
        // Führe die Speicherlogik für jeden einzelnen Reim aus
        rhymesToSave.forEach(rhyme => {
            const existingGroup = currentLibrary.find(item => item.type === 'rhyme_lesson_group' && (item as RhymeLessonGroup).targetWord === lessonWord) as RhymeLessonGroup | undefined;

            if (existingGroup) {
                // FALL 1: Gruppe existiert -> Füge neuen Reim hinzu, falls er noch nicht existiert
                if (!existingGroup.rhymes.some(r => r.rhymingWord === rhyme)) {
                    const newRhyme: RhymeLessonItem = { id: `subrhyme-${Date.now()}`, rhymingWord: rhyme };
                    const updatedGroup = { ...existingGroup, rhymes: [...existingGroup.rhymes, newRhyme] };
                    currentLibrary = currentLibrary.map(item => item.id === existingGroup.id ? updatedGroup : item);
                }
            } else {
                // FALL 2: Gruppe existiert nicht -> Erstelle neue Gruppe
                const analysis = getPhoneticBreakdown(lessonWord);
                const newGroup: RhymeLessonGroup = {
                    id: `rlg-${Date.now()}`,
                    type: 'rhyme_lesson_group',
                    content: `Reim-Lektionen für '${lessonWord}'`,
                    targetWord: lessonWord,
                    vowelSequence: analysis.vowelSequence,
                    syllableCount: analysis.syllableCount,
                    rhymes: [{ id: `subrhyme-${Date.now()}`, rhymingWord: rhyme }]
                };
                currentLibrary.push(newGroup);
            }
        });

        return currentLibrary;

    }, "Reim-Lektion(en) erfolgreich gespeichert.");

    // Felder nach dem Speichern leeren
    setLessonWord('');
    setLessonRhyme('');
};

const handleEditCategoryTitle = (categoryId: string, newTitle: string) => {
    if (!newTitle.trim()) {
        showStatus("Kategorie-Titel darf nicht leer sein.", true);
        // Optional: Reset auf alten Titel, hier aber erstmal einfach
        setIsEditingCategoryTitle(null);
        return;
    }
    updateProfileLibrary(lib => lib.map(item =>
        (item.id === categoryId && item.type === 'rule_category') ? { ...item, categoryTitle: newTitle } : item
    ));
    setIsEditingCategoryTitle(null);
};

const handleEditRuleTitle = (categoryId: string, ruleId: string, newTitle: string) => {
    if (!newTitle.trim()) {
        showStatus("Regel-Titel darf nicht leer sein.", true);
        setIsEditingRuleTitle(null);
        return;
    }
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === categoryId && item.type === 'rule_category') {
            const updatedRules = item.rules.map(r => r.id === ruleId ? { ...r, title: newTitle } : r);
            return { ...item, rules: updatedRules };
        }
        return item;
    }));
    setIsEditingRuleTitle(null);
};

const handleEditRuleDefinition = (categoryId: string, ruleId: string, newDefinition: string) => {
    if (!newDefinition.trim()) {
        showStatus("Regel-Definition darf nicht leer sein.", true);
        setIsEditingRuleDefinition(null);
        return;
    }
    updateProfileLibrary(lib => lib.map(item => {
        if (item.id === categoryId && item.type === 'rule_category') {
            const updatedRules = item.rules.map(r => r.id === ruleId ? { ...r, definition: newDefinition } : r);
            return { ...item, rules: updatedRules };
        }
        return item;
    }));
    setIsEditingRuleDefinition(null);
};

const getUnifiedKnowledgeBase = useCallback(() => {
    if (!activeProfile) return "Keine Künstler-DNA verfügbar.";

    // Lade ALLE relevanten Teile aus der Bibliothek des aktiven Profils
    const ruleCategories = activeProfile.library.filter(item => item.type === 'rule_category') as RuleCategory[];
    const styles = activeProfile.library.filter(item => item.type === 'style');
    const techniques = activeProfile.library.filter(item => item.type === 'technique');
    const rhymeLessons = activeProfile.library.filter(item => item.type === 'rhyme_lesson');
    const userExplanations = activeProfile.library
        .filter(item => item.type === 'lyric' && (item.userEmphasisExplanation || item.userRhymeFlowExplanation))
        .map(item => item.userEmphasisExplanation || item.userRhymeFlowExplanation);

    let knowledge = "### KÜNSTLER-DNA ###\n" +
                    "Dies ist das gesamte Wissen über den Künstler. Es besteht aus explizit gelernten Regeln, Stil- und Technik-Definitionen sowie Lektionen.\n\n";

    // Baue die Wissensbasis dynamisch aus den gelernten Regeln auf
    if (ruleCategories.length > 0) {
        knowledge += "### GELERNTES REGELWERK (Absolute Wahrheit) ###\n";
        ruleCategories.forEach(category => {
            knowledge += `**Kategorie: ${category.categoryTitle}**\n`;
            category.rules.forEach(rule => {
                knowledge += `- **Regel: ${rule.title}**\n  - Definition: ${rule.definition}\n`;
            });
        });
        knowledge += "\n";
    } else {
        knowledge += "### GELERNTES REGELWERK ###\n- Es wurden noch keine spezifischen Regeln durch den KI-Trainer definiert.\n\n";
    }

    if (styles.length > 0) {
        knowledge += "### KÜNSTLERISCHER STIL & THEMEN ###\n" + styles.map(s => `- ${s.content}`).join("\n") + "\n\n";
    }
    if (techniques.length > 0) {
        knowledge += "### TECHNISCHE FÄHIGKEITEN & VORLIEBEN ###\n" + techniques.map(t => `- ${t.content}`).join("\n") + "\n\n";
    }
    if (rhymeLessons.length > 0) {
        knowledge += "### EXPLIZITE REIM-LEKTIONEN (Nutzer-definiert) ###\n" + rhymeLessons.map(l => `- ${l.content}`).join("\n") + "\n\n";
    }
    if (userExplanations.length > 0) {
        knowledge += "### NUTZER-KORRIGIERTE EINSICHTEN ###\n" + userExplanations.filter(Boolean).map(e => `- ${e}`).join("\n") + "\n\n";
    }

    // Füge Fallback-Regeln hinzu, falls keine spezifischen Regeln existieren
    if (ruleCategories.length === 0) {
        knowledge += "### GRUNDLEGENDE FALLBACK-REGELN ###\n" +
                     "- REGEL: Semantische Plausibilität ist Pflicht. Die Reimphrase muss logisch und realweltlich Sinn ergeben.\n" +
                     "- REGEL: Identische Reime (z.B. 'Haus' auf 'Haus') sind unerwünscht.\n\n";
    }

    return knowledge;
}, [activeProfile]);

const getPhoneticBreakdown = (word: string): { syllableCount: number; vowelSequence: string } => {
    let tempWord = word.toLowerCase();

    if (tempWord.endsWith('er')) {
        tempWord = tempWord.slice(0, -2) + 'a';
    }    
    
    // Schritt 1: Dehnungs-h nach einem Vokal entfernen
    const vowelsWithH = ['ah', 'eh', 'ih', 'oh', 'uh', 'äh', 'öh', 'üh'];
    vowelsWithH.forEach(vh => {
        tempWord = tempWord.replace(new RegExp(vh, 'g'), vh[0]);
        tempWord = tempWord.replace(/([aouäöü])h/g, '$1'); // Dehnungs-h entfernen (z.B. "mahlen" -> "malen")
        tempWord = tempWord.replace(/ch/g, 'X'); // ch als eigener Konsonant, um nicht 'c' und 'h' einzeln zu werten
        tempWord = tempWord.replace(/sch/g, 'Y'); // sch als eigener Konsonant
        tempWord = tempWord.replace(/ck/g, 'k'); // ck zu k
    });

    // Schritt 2: Diphthonge und Vokale in der richtigen Reihenfolge suchen
    const vowelsAndDiphthongs = ['au', 'eu', 'äu', 'ei', 'ai', 'ey', 'ay', 'ie', 'a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü'];
    const vowelSequenceList: string[] = [];
    let i = 0;
    while (i < tempWord.length) {
        let found = false;
        for (const vOrD of vowelsAndDiphthongs) {
            if (tempWord.substring(i).startsWith(vOrD)) {
                vowelSequenceList.push(vOrD);
                i += vOrD.length;
                found = true;
                break;
            }
        }
        if (!found) {
            i++;
        }
    }
         return {
        syllableCount: vowelSequenceList.length,
        vowelSequence: vowelSequenceList.join('-')
    };
};
    // === VERARBEITUNGSKETTEN ===
    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            handleInitialAudioAnalysis(file, file.type);
        }
    };

    const handleInitialAudioAnalysis = async (blob: Blob, mimeType: string) => {
        setLoadingMessage('Transkribiere Audio & führe Voranalyse durch...');
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = async () => {
            const base64Audio = (reader.result as string).split(',')[1];
            try {
                const response = await fetch(`${BACKEND_URL}/api/analyze-audio`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ base64Audio, mimeType })
                });
                if (!response.ok) { const err = await response.json(); throw new Error(err.error || 'Vor-Analyse fehlgeschlagen'); }
                const result: PendingAnalysis = await response.json();
                setPendingAnalysis({ ...result, audioBlob: blob });
                setEditableTitle(result.title);
                setIsTitleModalOpen(true);
            } catch (error: any) {
                showStatus(error.message, true);
            } finally {
                setLoadingMessage('');
            }
        };
    };

    const handleConfirmTitle = () => {
        if (!pendingAnalysis) return;
        setPendingAnalysis(prev => prev ? { ...prev, title: editableTitle } : null);
        setIsTitleModalOpen(false);
        setEditableLyrics(pendingAnalysis.lyrics);
        setIsLyricEditorModalOpen(true);
    };

    const handleConfirmEditedLyrics = async () => {
        if (!pendingAnalysis || !activeProfileId) return;
        setIsLyricEditorModalOpen(false);
        setLoadingMessage('Starte umfassende Tiefenanalyse...');
        try {
            const payload = { lyrics: editableLyrics, knowledgeBase: getUnifiedKnowledgeBase(), hasAudio: !!pendingAnalysis.audioBlob };
            const response = await fetch(`${BACKEND_URL}/api/deep-analyze`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
            });
            if (!response.ok) { const err = await response.json(); throw new Error(err.error || 'Tiefenanalyse fehlgeschlagen'); }
            const result = await response.json();

            const newLyricItem: LibraryItem = {
                id: `lyric-${Date.now()}`, type: 'lyric', title: pendingAnalysis.title,
                content: result.formattedLyrics, emphasisPattern: result.emphasisPattern, rhymeFlowPattern: result.rhymeFlowPattern
            };
            const newStyleItems: LibraryItem[] = (result.characterTraits || []).map((trait: string, i: number) => ({ id: `style-${Date.now()}-${i}`, type: 'style', content: trait, sourceLyricId: newLyricItem.id }));
            const newTechniqueItems: LibraryItem[] = (result.technicalSkills || []).map((skill: string, i: number) => ({ id: `technique-${Date.now()}-${i}`, type: 'technique', content: skill, sourceLyricId: newLyricItem.id }));

            updateProfileLibrary(lib => [...lib, newLyricItem, ...newStyleItems, ...newTechniqueItems]);
            showStatus('Analyse erfolgreich abgeschlossen!', false);
            setCurrentView('manage_library');
        } catch (error: any) {
            showStatus(`Fehler bei der Analyse: ${error.message}`, true);
        } finally {
            setLoadingMessage('');
            setPendingAnalysis(null);
        }
    };

const handleSendTrainerMessage = async (text: string) => {
    if (!text.trim()) return;

    const userMessage = { text, isUser: true };
    const newMessages = [...trainerMessages, userMessage];
    setTrainerMessages(newMessages);
    setIsTrainerReplying(true);

    try {
        const payload = {
            messages: newMessages,
            knowledgeBase: getUnifiedKnowledgeBase()
        };
        const response = await fetch(`${BACKEND_URL}/api/trainer-chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Chat-Antwort fehlgeschlagen');
        }
        const result = await response.json();

        const aiMessage = { text: result.reply, isUser: false };
        setTrainerMessages(prev => [...prev, aiMessage]);

        // === KORRIGIERTER BLOCK ===
        // Wir prüfen, ob irgendein Lern-Objekt vorhanden ist und fügen es hinzu.
        if (result.learning) {
            const newLearnedItem = result.learning as LibraryItem | RuleCategory; // Akzeptiert beide Typen
            
            // Erstelle eine passende Erfolgsmeldung
            let successMessage = "Neue Regel zur DNA hinzugefügt!";
            if (newLearnedItem.type === 'rule_category') {
                successMessage = `Neue Regelkategorie '${(newLearnedItem as RuleCategory).categoryTitle}' zur DNA hinzugefügt!`;
            } else if (newLearnedItem.type === 'rhyme_rule') {
                // Annahme: Das 'rhyme_rule' Objekt hat ein 'syllableCount' Feld
                // Wir müssen TypeScript hier sagen, dass es existiert.
                const syllableCount = (newLearnedItem as any).syllableCount;
                successMessage = `Neue ${syllableCount}-Silben-Reimregel zur DNA hinzugefügt!`;
            }

            updateProfileLibrary(
                lib => [...lib, newLearnedItem],
                successMessage
            );
        }
    } catch (error: any) {
        showStatus(error.message, true);
    } finally {
        setIsTrainerReplying(false);
    }
};

    // === RENDER-FUNKTIONEN ===
    const renderStartScreen = () => (
        <div className={`start-screen-container ${isStarting ? 'fading-out' : ''}`}>
            <div className="start-screen-content">
                <h1>Gallant's Lyric <span>Machine</span></h1>
                <p>Deine persönliche KI-Songwriting-Maschine.</p>
                <button className="start-button" onClick={(e) => {
                  const button = e.currentTarget;
                  const circle = document.createElement("span");
                  const diameter = Math.max(button.clientWidth, button.clientHeight);
                  const radius = diameter / 2;
                  circle.style.width = circle.style.height = `${diameter}px`;
                  circle.style.left = `${e.clientX - button.offsetLeft - radius}px`;
                  circle.style.top = `${e.clientY - button.offsetTop - radius}px`;
                  circle.classList.add("ripple");
                  const ripple = button.getElementsByClassName("ripple")[0];
                  if (ripple) { ripple.remove(); }
                  button.appendChild(circle);
                  setIsStarting(true);
                  setTimeout(() => {
                      setCurrentView('transition');
                      setTimeout(() => {
                          setCurrentView('analyze');
                      }, 2400);
                  }, 500);
                }}>Start</button>
            </div>
        </div>
    );

    const renderTransitionScreen = () => (
        <div className="transition-screen">
            <div className="gallant-text">Made by Gallant</div>
        </div>
    );

    const renderModals = () => (
        <>
            {isTitleModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-content">
                        <h3>Titel bestätigen</h3>
                        <p className="panel-description">Die KI schlägt diesen Titel vor. Du kannst ihn hier anpassen.</p>
                        <input type="text" value={editableTitle} onChange={e => setEditableTitle(e.target.value)} />
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsTitleModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleConfirmTitle}>Weiter</button>
                        </div>
                    </div>
                </div>
            )}
            {isLyricEditorModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-content wide">
                        <h3>Transkription bearbeiten</h3>
                        <p className="panel-description">Korrigiere hier den transkribierten Text, bevor die endgültige Tiefenanalyse gestartet wird.</p>
                        <textarea value={editableLyrics} onChange={e => setEditableLyrics(e.target.value)} className="lyrics-editor"/>
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsLyricEditorModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleConfirmEditedLyrics}>Analyse starten</button>
                        </div>
                    </div>
                </div>
            )}
            {isCreateProfileModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-content">
                        <h3>Neues Profil erstellen</h3>
                        <input type="text" value={newProfileName} onChange={e => setNewProfileName(e.target.value)} placeholder="Name des neuen Profils"/>
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsCreateProfileModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleCreateProfile}>Erstellen</button>
                        </div>
                    </div>
                </div>
            )}
            {isEditModalOpen && editingItem && (
                 <div className="modal-overlay">
                    <div className="modal-content wide">
                        <h3>"{editingItem.title || editingItem.type}" bearbeiten</h3>
                        <textarea value={editedContent} onChange={(e) => setEditedContent(e.target.value)} className="lyrics-editor"/>
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsEditModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleSaveEdit}>Speichern</button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );

const renderAppHeader = () => (
    <header className="app-header">
        <h1>Gallant's Lyric <span>Machine</span></h1>
        {activeProfile && (
            <div className="profile-selector header-profile-selector">
                <span>Aktives Profil:</span>
                <select value={activeProfileId ?? ''} onChange={e => setActiveProfileId(e.target.value)}>
                    {Object.values(profiles).map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
                <button
                    className="add-profile-button"
                    title="Neues Profil erstellen"
                    onClick={() => setIsCreateProfileModalOpen(true)}
                >
                    {icons.add}
                </button>
            </div>
        )}
    </header>
);

    const renderSideMenu = () => {
        const menuItems: { key: View, label: string, icon: JSX.Element }[] = [
            { key: 'analyze', label: 'Song analysieren', icon: icons.analyze },
            { key: 'define_style', label: 'Stil definieren', icon: icons.style },
            { key: 'define_technique', label: 'Technik definieren', icon: icons.technique },
            { key: 'trainer', label: 'KI-Trainer', icon: icons.technique },
            { key: 'rhyme_machine', label: 'Rhyme Machine', icon: icons.rhyme },
            { key: 'write_song', label: 'Song schreiben', icon: icons.write },
            { key: 'manage_library', label: 'Bibliothek verwalten', icon: icons.library }
        ];

        return (
            <aside className="side-menu">
                <div className="menu-header">Menü</div>
                <nav>
                    {menuItems.map(item => (
                        <button key={item.key} className={`menu-button ${currentView === item.key ? 'active' : ''}`} onClick={() => setCurrentView(item.key)}>
                            {item.icon}
                            <span>{item.label}</span>
                        </button>
                    ))}
                </nav>
            </aside>
        );
    };

const renderTrainerView = () => (
    <Panel title="KI-Trainer" icon={icons.technique} description="Führe ein Gespräch mit der KI, um ihr neue Regeln beizubringen. Jede gelernte Lektion wird permanent in der Künstler-DNA des aktiven Profils gespeichert.">
        <div className="chat-container">
            <div className="chat-messages">
                {trainerMessages.map((msg, index) => (
                    <div key={index} className={`chat-bubble ${msg.isUser ? 'user' : 'ai'}`}>
                        <div className="author">
                            {msg.isUser ? (activeProfile?.name || 'Benutzer') : 'KI-Trainer'}
                        </div>
                        {msg.text}
                    </div>
                ))}
                {isTrainerReplying && (
                    <div className="chat-bubble ai">
                        <div className="author">KI-Trainer</div>
                        <div className="typing-indicator">
                            <div className="typing-dot"></div>
                            <div className="typing-dot"></div>
                            <div className="typing-dot"></div>
                        </div>
                    </div>
                )}
            </div>
            <ChatInput onSend={handleSendTrainerMessage} isReplying={isTrainerReplying} />
        </div>
    </Panel>
);

const renderAnalyzeView = () => (
    <Panel title="Bibliothek füllen" icon={icons.analyze} description="Analysiere eine oder mehrere Audiodateien, um Songs inklusive Stil, Technik und Rhythmus automatisch in deine Bibliothek aufzunehmen.">
        {/* Dieser Inhalt ist jetzt KORREKT innerhalb des Panels */}
        <div className="main-action-container">
            <button className="main-action-button" onClick={() => fileInputRef.current?.click()}>Songs analysieren (Audio-Dateien)</button>
            <button className="record-button" title="Song direkt aufnehmen"><span className="record-dot"></span></button>
        </div>
        <input type="file" ref={fileInputRef} onChange={handleFileUpload} style={{ display: 'none' }} accept="audio/*" />
        <div className="divider">ODER</div>
        <div className="text-input-section">
            <p>Füge hier Text manuell ein, um ihn zu analysieren.</p>
            {/* Hier verwenden wir die neuen Memoized Komponenten */}
            <MemoizedInput
                placeholder="Songtitel (optional)"
                value={manualTitle}
                onChange={e => setManualTitle(e.target.value)}
             />

            <MemoizedTextarea
                placeholder="Songtext hier einfügen..."
                value={manualLyrics}
                onChange={e => setManualLyrics(e.target.value)}
            />
            <button className="secondary-action-button">Text analysieren</button>
        </div>
    </Panel>
);

const renderLibraryView = () => {
    // Vereinfachte Liste der Tabs für eine saubere Anzeige
    const tabs: { key: LibraryTab, label: string }[] = [
        { key: 'learned_lyrics', label: 'Gelernte Lyrics' },
        { key: 'learned_rules', label: 'Gelernte Regeln' },
        { key: 'rhyme_lessons', label: 'Gelernte Reime' },
        { key: 'generated_lyrics', label: 'Generierte Lyrics' },
        { key: 'character', label: 'Character / Stil' },
        { key: 'technique', label: 'Technik' },
    ];

    const getTabCount = (tabKey: LibraryTab) => {
        if (!activeProfile) return 0;
        const lib = activeProfile.library;
        switch(tabKey) {
            case 'learned_lyrics': return lib.filter(i => i.type === 'lyric').length;
            case 'learned_rules': return lib.filter(i => i.type === 'rule_category').length;
            case 'rhyme_lessons': return lib.filter(i => i.type === 'rhyme_lesson_group').length;
            case 'generated_lyrics': return lib.filter(i => i.type === 'generated_lyric').length;
            case 'character': return lib.filter(i => i.type === 'style').length;
            case 'technique': return lib.filter(i => i.type === 'technique').length;
            default: return 0;
        }
    };

    const filterItemsForTab = (tabKey: LibraryTab) => {
        if (!activeProfile) return [];
        const lib = activeProfile.library;
        switch(tabKey) {
            case 'learned_lyrics': return lib.filter(i => i.type === 'lyric');
            case 'learned_rules': return lib.filter(i => i.type === 'rule_category');
            case 'rhyme_lessons': return lib.filter(i => i.type === 'rhyme_lesson_group');
            case 'generated_lyrics': return lib.filter(i => i.type === 'generated_lyric');
            case 'character': return lib.filter(i => i.type === 'style');
            case 'technique': return lib.filter(i => i.type === 'technique');
            default: return [];
        }
    };

    const filteredItems = filterItemsForTab(libraryTab);

    return (
        <>
             {/* NEUES MODAL FÜR REIM-BEARBEITUNG */}
        {isSubRhymeEditModalOpen && editingSubRhyme && (
            <div className="modal-overlay">
                <div className="modal-content">
                    <h3>Reim bearbeiten</h3>
                    <input type="text" value={editedSubRhymeContent} onChange={e => setEditedSubRhymeContent(e.target.value)} autoFocus/>
                    <div className="modal-actions">
                        <button className="secondary-action-button" onClick={() => setIsSubRhymeEditModalOpen(false)}>Abbrechen</button>
                        <button className="main-action-button" onClick={handleSaveSubRhymeEdit}>Speichern</button>
                    </div>
                </div>
            </div>
        )}

            {isRuleEditModalOpen && editingRule && (
                <div className="modal-overlay">
                    <div className="modal-content wide">
                        <h3>"{editingRule.rule.title}" bearbeiten</h3>
                        <textarea
                            value={editedRuleContent}
                            onChange={(e) => setEditedRuleContent(e.target.value)}
                            className="lyrics-editor"
                        />
                        <div className="modal-actions">
                            <button className="secondary-action-button" onClick={() => setIsRuleEditModalOpen(false)}>Abbrechen</button>
                            <button className="main-action-button" onClick={handleSaveRuleEdit}>Speichern</button>
                        </div>
                    </div>
                </div>
            )}
            <Panel title="Bibliothek verwalten" icon={icons.library} description="Hier werden alle gelernten Elemente deiner Künstler-DNA gespeichert und verwaltet.">
                 <div className="library-header">
                    <div className="library-tabs">
                        {tabs.map(tab => (
                            <button key={tab.key} className={`tab-button ${libraryTab === tab.key ? 'active' : ''}`} onClick={() => setLibraryTab(tab.key)}>
                                {tab.label} ({getTabCount(tab.key)})
                            </button>
                        ))}
                    </div>
                 </div>
                 <div className="library-content">
                    {filteredItems.length === 0 && <p>Diese Sektion ist leer.</p>}

                    {libraryTab === 'learned_rules' && filteredItems.map(item => {
                        const category = item as RuleCategory;
                        return (
                            <div key={category.id} className="accordion-item">
                                <div className="accordion-header" onClick={() => setOpenRuleCategoryId(openRuleCategoryId === category.id ? null : category.id)}>
                                    <span onDoubleClick={() => setIsEditingCategoryTitle({ id: category.id, title: category.categoryTitle })}>{category.categoryTitle}</span>
                                    <div className={`accordion-icon ${openRuleCategoryId === category.id ? 'open' : ''}`}>{icons.chevronDown}</div>
                                </div>
                                {openRuleCategoryId === category.id && (
                                    <div className="accordion-content">
                                        {category.rules?.map(rule => (
                                            <div key={rule.id} className="simple-library-item rule-item">
                                                <strong onDoubleClick={() => setIsEditingRuleTitle({ categoryId: category.id, ruleId: rule.id, title: rule.title })}>{rule.title}</strong>
                                                <p onDoubleClick={() => setIsEditingRuleDefinition({ categoryId: category.id, ruleId: rule.id, definition: rule.definition })}>{rule.definition}</p>
                                                <div className="item-actions">
                                                    <button className="action-button" onClick={() => handleStartRuleEdit(category.id, rule)}>Bearbeiten</button>
                                                    <button className="action-button delete" onClick={() => handleDeleteRule(category.id, rule.id)}>Löschen</button>
                                                </div>
                                            </div>
                                        ))}
                                        <div className="item-actions top">
                                            <button className="action-button delete" onClick={() => handleDeleteRuleCategory(category.id)}>Kategorie löschen</button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}

                    {(libraryTab === 'learned_lyrics' || libraryTab === 'generated_lyrics') && filteredItems.map(item => {
                         const lyric = item as LibraryItem;
                         return (
                            <div key={lyric.id} className="accordion-item">
                                <div className="accordion-header" onClick={() => setOpenLyricId(openLyricId === lyric.id ? null : lyric.id)}>
                                    <span>{lyric.title || 'Unbenannter Song'}</span>
                                    <div className={`accordion-icon ${openLyricId === lyric.id ? 'open' : ''}`}>{icons.chevronDown}</div>
                                </div>
                                {openLyricId === lyric.id && (
                                    <div className="accordion-content">
                                        <div className="item-actions top">
                                            <button className="action-button" onClick={() => handleStartEditing(lyric)}>Text bearbeiten</button>
                                            <button className="action-button delete" onClick={() => handleDeleteItem(lyric.id)}>Song löschen</button>
                                        </div>
                                        <h4>Originaltext</h4><pre>{lyric.content}</pre>
                                    </div>
                                )}
                            </div>
                         );
                    })}

                    {libraryTab === 'rhyme_lessons' && (() => {
    const lessonGroups = filteredItems as RhymeLessonGroup[];

    const groupedByVowelSequence = lessonGroups.reduce((acc, group) => {
        const key = group.vowelSequence || 'Unbekannt';
        if (!acc[key]) { acc[key] = []; }
        acc[key].push(group);
        return acc;
    }, {} as Record<string, RhymeLessonGroup[]>);

    return Object.entries(groupedByVowelSequence).map(([sequence, groupsInSequence]) => (
        <div key={sequence} className="accordion-item">
            <div className="accordion-header level-1" onClick={() => setOpenRuleCategoryId(openRuleCategoryId === sequence ? null : sequence)}>
                <span>Vokalfolge: {sequence}</span>
                <div className={`accordion-icon ${openRuleCategoryId === sequence ? 'open' : ''}`}>{icons.chevronDown}</div>
            </div>
            {openRuleCategoryId === sequence && (
                <div className="accordion-content">
                    {groupsInSequence.map(group => (
                        <div key={group.id} className="accordion-item nested">
                            <div className="accordion-header level-2" onClick={() => setOpenLyricId(openLyricId === group.id ? null : group.id)}>
                                <span>Ausgangswort: <strong>{group.targetWord}</strong> ({group.rhymes?.length || 0} Reime)</span>
                                <div className={`accordion-icon ${openLyricId === group.id ? 'open' : ''}`}>{icons.chevronDown}</div>
                            </div>
                            {openLyricId === group.id && (
                                <div className="accordion-content">
                                    {group.rhymes?.map((rhyme) => (
                                        <div key={rhyme.id} className="simple-library-item rule-item">
                                            <p>{rhyme.rhymingWord}</p>
                                            <div className="item-actions">
                                                <button className="action-button" onClick={() => handleStartSubRhymeEdit(group.id, rhyme)}>Bearbeiten</button>
                                                <button className="action-button delete" onClick={() => handleDeleteSubRhyme(group.id, rhyme.id)}>Löschen</button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    ));
})()}

                    {(libraryTab === 'character' || libraryTab === 'technique') && groupedItems && groupedItems.map(group => (
                        <div key={group.lyric.id} className="grouped-item-container">
                            <h4>Aus "{group.lyric.title || 'Unbenannter Song'}"</h4>
                            <div className="simple-item-list">
                                {group.items.map(item => (
                                    <div key={item.id} className="simple-library-item">
                                        <p>{item.content}</p>
                                        <div className="item-actions">
                                            <button className="action-button" onClick={() => handleStartEditing(item as LibraryItem)}>Bearbeiten</button>
                                            <button className="action-button delete" onClick={() => handleDeleteItem(item.id)}>Löschen</button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                 </div>
            </Panel>
        </>
    );
};

    const renderWriteSongView = () => {
        const styles = activeProfile?.library.filter(i => i.type === 'style') ?? [];
        const techniques = activeProfile?.library.filter(i => i.type === 'technique') ?? [];

        return (
            <Panel title="Song schreiben lassen" icon={icons.write} description="Wähle die Zutaten für deinen Song: Profil, Beat, Thema und die gewünschten Stil-Elemente.">
                <div className="form-grid">
                    <div className="form-group span-2">
                        <label>Song-Fusion (Optional)</label>
                        <select value={songFusion} onChange={e => setSongFusion(e.target.value)}>
                            <option>Wähle 1 oder mehr Songs aus deiner Bibliothek...</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Character / Stil</label>
                        <select value={songStyle} onChange={e => setSongStyle(e.target.value)}>
                            <option>Stile auswählen ({styles.length})</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Technik / Skills</label>
                        <select value={songTechnique} onChange={e => setSongTechnique(e.target.value)}>
                            <option>Techniken auswählen ({techniques.length})</option>
                        </select>
                    </div>
                    <div className="form-group span-2">
                        <label>Beat-Beschreibung (z.B. "melancholischer Lo-Fi Beat")</label>
                        <textarea
                            rows={3}
                            value={songBeatDescription}
                            onChange={e => setSongBeatDescription(e.target.value)}
                        ></textarea>
                    </div>
                     <div className="form-group">
                        <label>BPM (optional)</label>
                        <input
                            type="text"
                            value={songBPM}
                            onChange={e => setSongBPM(e.target.value)}
                        />
                    </div>
                     <div className="form-group">
                        <label>Tonart (optional)</label>
                        <input
                            type="text"
                            value={songKey}
                            onChange={e => setSongKey(e.target.value)}
                        />
                    </div>
                    <div className="form-group span-2">
                         <button className="secondary-action-button full-width">Beat-Datei analysieren</button>
                    </div>
                     <div className="form-group">
                        <label>Performance-Stil</label>
                        <select value={songPerformanceStyle} onChange={e => setSongPerformanceStyle(e.target.value)}>
                            <option>Gerappt</option>
                            <option>Gesungen</option>
                        </select>
                    </div>
                     <div className="form-group">
                        <label>Thema des Songs (z.B. "Eine lange Nacht")</label>
                        <input
                            type="text"
                            value={songTopic}
                            onChange={e => setSongTopic(e.target.value)}
                        />
                    </div>
                </div>
                <button className="main-action-button full-width">Songtext generieren</button>
                <div className="output-container">
                    <p className="placeholder">Dein generierter Songtext erscheint hier...</p>
                </div>
            </Panel>
        );
    };

    const renderDefineStyleView = () => (
         <Panel title="Stil definieren" icon={icons.style} description="Füge hier dauerhafte Stilanweisungen zum aktuell ausgewählten Profil hinzu (z.B. Themen, Stimmung, Bildsprache). Diese werden Teil der 'Künstler-DNA'.">
            <textarea
                placeholder="z.B. 'Texte sind oft melancholisch mit urbaner Bildsprache'."
                value={styleInput}
                onChange={e => setStyleInput(e.target.value)}
            />
            <button className="secondary-action-button">Anweisung zur DNA hinzufügen</button>
        </Panel>
    );

    const renderDefineTechniqueView = () => (
         <Panel title="Technik definieren" icon={icons.technique} description="Füge hier dauerhafte technische Anweisungen zum aktuell ausgewählten Profil hinzu (z.B. Reimschemata, Flow-Muster). Diese werden Teil der 'Künstler-DNA'.">
            <textarea
                placeholder="z.B. 'Bevorzuge zweisilbige Reime mit dem Reimschema AABB'."
                value={techniqueInput}
                onChange={e => setTechniqueInput(e.target.value)}
            />
            <button className="secondary-action-button">Anweisung zur DNA hinzufügen</button>
        </Panel>
    );

const renderRhymeMachineView = () => (
    <Panel title="Rhyme Machine" icon={icons.rhyme} description="Die KI findet Reime basierend auf deinen gelernten Regeln.">
        <div className="sub-panel">
            <h3>1. Wort eingeben & Reime finden</h3>
            <div className="rhyme-step-container">
                <input
                    type="text"
                    placeholder="z.B. Geschichte"
                    value={rhymeInput}
                    onChange={e => {
                        setRhymeInput(e.target.value);
                        // Die Analyse wird jetzt direkt von der KI durchgeführt, daher keine lokale Anzeige mehr
                        setRhymeResults([]);
                        setSearchPerformed(false);
                    }}
                />
                <button
                    className="main-action-button"
                    onClick={handleFindRhymesWithAnalysis}
                    disabled={isFindingRhymes}
                >
                    {isFindingRhymes ? 'Suche...' : 'Reime finden'}
                </button>
            </div>
            <div className="output-container rhyme-results">
                {isFindingRhymes ? (
                    <div className="typing-indicator" style={{ marginTop: '20px' }}>
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                    </div>
                ) : rhymeResults.length > 0 ? (
                    rhymeResults.map((result, index) => (
                        <div key={index} className="rhyme-result-item">
                            <h4>{result.rhyme}</h4>
                            <p>{result.explanation}</p>
                        </div>
                    ))
                ) : searchPerformed ? (
                    <p className="placeholder">Keine passenden Reime gefunden, die den Regeln entsprechen.</p>
                ) : (
                    <p className="placeholder">Deine Reim-Ergebnisse erscheinen hier...</p>
                )}
            </div>
        </div>
        <div className="sub-panel">
            <h3>Mehrsilben-Reim-Generator</h3>
            <p>Gib eine ganze Zeile ein, um phonetisch passende Folgezeilen mit komplexen, mehrsilbigen und thematisch sinnvollen Reimen zu finden.</p>
            <input
                type="text"
                placeholder="z.B. Kicken auf dem Bolzer bis zum Sonnenuntergang."
                value={multiRhymeInput}
                onChange={e => setMultiRhymeInput(e.target.value)}
            />
            <button className="secondary-action-button">Komplexe Reime finden</button>
        </div>
        <div className="sub-panel">
            <h3>Bringe der KI deine Reime bei</h3>
            <p>Je mehr Reime du hier eingibst, desto besser versteht die KI deine persönliche Reimlogik. Diese Lektionen fließen in die gesamte Künstler-DNA ein.</p>
            <div className="rhyme-lesson-input">
                <input
                    type="text"
                    placeholder="Wort / Phrase"
                    value={lessonWord}
                    onChange={e => setLessonWord(e.target.value)}
                />
                <span>reimt sich auf</span>
                <input
                    type="text"
                    placeholder="Dein Reim"
                    value={lessonRhyme}
                    onChange={e => setLessonRhyme(e.target.value)}
                />
            </div>
            <button className="secondary-action-button" onClick={handleSaveRhymeLesson}>Lektion zur DNA hinzufügen</button>
        </div>
    </Panel>
);

    // Wir erstellen jetzt für jede Ansicht eine "memoized" Version.
    // Das bedeutet, sie wird nur neu gerendert, wenn sich ihre EIGENEN Eingabefelder ändern.

const renderViewContent = () => {
    switch(currentView) {
        case 'analyze': return renderAnalyzeView();
        case 'define_style': return renderDefineStyleView();
        case 'define_technique': return renderDefineTechniqueView();
        case 'trainer': return renderTrainerView();
        case 'rhyme_machine': return renderRhymeMachineView();
        case 'write_song': return renderWriteSongView();
        case 'manage_library': return renderLibraryView();
        default: return renderAnalyzeView();
    }
};

    // HAUPT-ROUTER der App
    if (currentView === 'start') return renderStartScreen();
    if (currentView === 'transition') return renderTransitionScreen();

    return (
        <div className="app-layout">
            {renderModals()}
            {loadingMessage && <div className="loading-overlay">{loadingMessage}</div>}
            {statusMessage.text && <div className={`status-message ${statusMessage.isError ? 'error' : 'success'}`}>{statusMessage.text}</div>}

            {renderSideMenu()}
            <main className="content-wrapper">
                {renderAppHeader()}
                <div className="view-content">
                    {renderViewContent()}
                </div>
            </main>
        </div>
    );
};

// === ANWENDUNG STARTEN ===
const container = document.getElementById('root');
if (container) {
    const root = createRoot(container);
    root.render(<React.StrictMode><App /></React.StrictMode>);
}