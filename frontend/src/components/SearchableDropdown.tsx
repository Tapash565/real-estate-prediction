import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search, X } from 'lucide-react';

interface SearchableDropdownProps {
    options: string[];
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
    label?: string;
}

const SearchableDropdown: React.FC<SearchableDropdownProps> = ({
    options,
    value,
    onChange,
    placeholder = "Select...",
    label
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const dropdownRef = useRef<HTMLDivElement>(null);

    const filteredOptions = options.filter(option =>
        option.toLowerCase().includes(searchTerm.toLowerCase())
    );

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className="input-group" ref={dropdownRef}>
            {label && <label>{label}</label>}
            <div className="relative">
                <div
                    className={`dropdown-trigger glass ${isOpen ? 'active' : ''}`}
                    onClick={() => setIsOpen(!isOpen)}
                >
                    <span className={value ? 'text-white' : 'text-gray-500'}>
                        {value || placeholder}
                    </span>
                    <ChevronDown size={18} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                </div>

                {isOpen && (
                    <div className="dropdown-menu glass animate-fadeIn">
                        <div className="p-2 border-b border-white/10 sticky top-0 bg-zinc-900/90 backdrop-blur-md">
                            <div className="relative flex items-center">
                                <Search size={14} className="absolute left-2 text-gray-400" />
                                <input
                                    type="text"
                                    className="w-full pl-8 pr-8 py-1.5 text-sm bg-white/5 border-none focus:ring-0 rounded-md"
                                    placeholder="Search..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    onClick={(e: React.MouseEvent) => e.stopPropagation()}
                                    autoFocus
                                />
                                {searchTerm && (
                                    <X
                                        size={14}
                                        className="absolute right-2 text-gray-400 cursor-pointer hover:text-white"
                                        onClick={(e: React.MouseEvent) => {
                                            e.stopPropagation();
                                            setSearchTerm("");
                                        }}
                                    />
                                )}
                            </div>
                        </div>
                        <div className="max-h-60 overflow-y-auto py-1">
                            {filteredOptions.length > 0 ? (
                                filteredOptions.map((option) => (
                                    <div
                                        key={option}
                                        className={`px-4 py-2 text-sm cursor-pointer hover:bg-white/10 transition-colors ${value === option ? 'bg-primary/20 text-primary' : ''
                                            }`}
                                        onClick={() => {
                                            onChange(option);
                                            setIsOpen(false);
                                            setSearchTerm("");
                                        }}
                                    >
                                        {option}
                                    </div>
                                ))
                            ) : (
                                <div className="px-4 py-3 text-sm text-gray-400 text-center">
                                    No results found
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SearchableDropdown;
